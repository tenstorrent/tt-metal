# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import fnmatch
import os
import shlex
import socket
import subprocess
import sys
import weakref

import cloudpickle
import yaml

from vllm.config import ParallelConfig, VllmConfig
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.network_utils import get_ip
from vllm.utils.system_utils import kill_process_tree
from vllm.v1.engine.utils import CoreEngine, CoreEngineLauncher, EngineLaunchPlan
from vllm.v1.executor.abstract import UniProcExecutor
from vllm_tt_plugin.config import get_tt_config
from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)


class TTLaunchPlan(EngineLaunchPlan):
    rank_binding_file: str | None = None


class TTCoreEngineLauncher(CoreEngineLauncher):
    def prepare_launch(self, vllm_config: VllmConfig) -> EngineLaunchPlan:
        rank_binding_file, non_device_dp_ranks = parse_tt_mpi_params(vllm_config)
        if rank_binding_file is None:
            return EngineLaunchPlan()

        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.data_parallel_master_ip
            == ParallelConfig.data_parallel_master_ip
        ):
            host_ip = get_ip()
            logger.info("Using host IP %s as TT data parallel address", host_ip)
            parallel_config.data_parallel_master_ip = host_ip

        vllm_config.parallel_config.data_parallel_size_local = len(non_device_dp_ranks)
        plan = TTLaunchPlan(
            remote_launched=True,
            non_device_dp_ranks=non_device_dp_ranks,
        )
        plan.rank_binding_file = rank_binding_file
        return plan

    def get_engines_to_handshake(
        self,
        vllm_config: VllmConfig,
        local_engine_count: int,
        local_start_index: int | None,
        dp_rank: int,
        dp_size: int,
        local_engines_only: bool,
        offline_mode: bool,
        plan: EngineLaunchPlan,
    ) -> list[CoreEngine]:
        if not plan.remote_launched:
            return super().get_engines_to_handshake(
                vllm_config,
                local_engine_count,
                local_start_index,
                dp_rank,
                dp_size,
                local_engines_only,
                offline_mode,
                plan,
            )
        non_device_dp_ranks = plan.non_device_dp_ranks or set()
        return [
            CoreEngine(index=i, local=(i in non_device_dp_ranks))
            for i in range(vllm_config.parallel_config.data_parallel_size)
        ]

    def launch_remote_engines(
        self,
        plan: EngineLaunchPlan,
        handshake_address: str,
        vllm_config: VllmConfig,
        log_stats: bool,
        cleanup_target: object,
    ) -> None:
        assert isinstance(plan, TTLaunchPlan)
        assert plan.rank_binding_file is not None
        # Launch must be done on the host with MPI rank 0 since we set that
        # process's DP rank to 0, and torch distributed uses DP rank 0 to bind
        # the TCP rendezvous endpoint.
        assert vllm_config.parallel_config.data_parallel_rank == 0, (
            "TT MPI must be launched from rank 0"
        )
        logger.info(
            "TT-MPI mixed launch: dp_size=%d local_count=%d "
            "non_device_ranks=%s handshake=%s",
            vllm_config.parallel_config.data_parallel_size,
            vllm_config.parallel_config.data_parallel_size_local,
            plan.non_device_dp_ranks,
            handshake_address,
        )
        tt_run_launch(
            handshake_address=handshake_address,
            vllm_config=vllm_config,
            rank_binding_file=plan.rank_binding_file,
            log_stats=log_stats,
            cleanup_target=cleanup_target,
        )


def _validate_launch_from_rank0_host(mpi_args: str, host_ip: str) -> None:
    if not mpi_args:
        return
    try:
        argv = shlex.split(mpi_args)
    except Exception:
        argv = []
    mapby_path = None
    # Parse --map-by to locate rankfile and ensure rank 0 host matches host_ip.
    for i, tok in enumerate(argv):
        if tok.startswith("--map-by"):
            if "=" in tok:
                value = tok.split("=", 1)[1]
            elif i + 1 < len(argv):
                value = argv[i + 1]
            else:
                value = ""
            if "file=" in value:
                mapby_path = value.split("file=", 1)[1].split(",", 1)[0].strip()
            if not mapby_path and value and os.path.isfile(value):
                mapby_path = value
            break
    if not (mapby_path and os.path.isfile(mapby_path)):
        return

    # Open MPI rankfiles use entries like `rank 0=<host> ...`.
    rank0_host = None
    try:
        with open(mapby_path) as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                tokens = stripped.split()
                if len(tokens) >= 2 and tokens[0] == "rank" and "=" in tokens[1]:
                    left, right = tokens[1].split("=", 1)
                    try:
                        rank_num = int(left)
                    except Exception:
                        rank_num = -1
                    if rank_num == 0:
                        rank0_host = right.strip()
                        break
    except Exception:
        rank0_host = None
    if not rank0_host:
        return

    # Accept either direct IP rankfiles or hostnames that resolve to host_ip.
    resolved_ips: set[str] = set()
    if all(c.isdigit() or c == "." for c in rank0_host):
        resolved_ips.add(rank0_host)
    try:
        info = socket.getaddrinfo(rank0_host, None, proto=socket.IPPROTO_TCP)
        for ai in info:
            resolved_ips.add(ai[4][0])
    except Exception as e:
        logger.warning(
            "Failed to resolve IP address for rank 0 host %s: %s", rank0_host, e
        )
        if not resolved_ips:
            return
    assert host_ip in resolved_ips, (
        f"MPI rank 0 host {rank0_host} from rankfile {mapby_path} "
        f"(resolves to {sorted(resolved_ips)}) does not match "
        f"launcher host IP {host_ip}."
    )
    logger.info("Validated launching from MPI rank 0 host %s", rank0_host)


def parse_tt_mpi_params(vllm_config: VllmConfig) -> tuple[str | None, set[int]]:
    parallel_config = vllm_config.parallel_config
    assert parallel_config.data_parallel_backend != "ray", (
        "TT does not support ray-based data parallel backend"
    )
    dp_size = parallel_config.data_parallel_size
    tt_config = get_tt_config(vllm_config)
    rank_binding_file = tt_config.get("rank_binding")
    non_device_dp_ranks: set[int] = set()
    if rank_binding_file:
        if not isinstance(rank_binding_file, str):
            raise RuntimeError(
                "TT plugin config key 'rank_binding' must be a non-empty string"
            )
        try:
            with open(rank_binding_file) as f:
                rb = yaml.safe_load(f)
            mpi_world = len(rb.get("rank_bindings", []))
        except Exception as e:
            raise RuntimeError(
                f"Failed to read rank binding '{rank_binding_file}': {e}"
            ) from e
        if mpi_world <= 0 or dp_size % mpi_world != 0:
            raise RuntimeError(
                f"data_parallel_size ({dp_size}) must be divisible by number "
                f"of device MPI ranks ({mpi_world})"
            )
        # Only the first DP rank in each MPI segment owns a TT device process.
        # The other DP ranks stay local and participate as non-device ranks.
        dp_size_per_mpi_rank = dp_size // mpi_world
        device_dp_ranks = {i * dp_size_per_mpi_rank for i in range(mpi_world)}
        non_device_dp_ranks = {i for i in range(dp_size) if i not in device_dp_ranks}
    return rank_binding_file, non_device_dp_ranks


def tt_run_launch(
    handshake_address: str,
    vllm_config: VllmConfig,
    rank_binding_file: str,
    log_stats: bool,
    cleanup_target: object,
) -> None:
    if not rank_binding_file:
        raise RuntimeError("rank_binding_file must be a non-empty string")

    tt_config = get_tt_config(vllm_config)
    mpi_args = tt_config.get("mpi_args", "")
    extra_ttrun_args = tt_config.get("extra_ttrun_args")
    cfg_dir = tt_config.get("config_pkl_dir")

    if not cfg_dir:
        raise RuntimeError(
            "TT plugin config key 'config_pkl_dir' is required for TT MPI launch"
        )
    if not os.path.isdir(cfg_dir):
        raise RuntimeError("TT plugin config key 'config_pkl_dir' must be a directory")

    host_ip = get_ip()
    parallel_config = vllm_config.parallel_config
    assert parallel_config.data_parallel_master_ip == host_ip, (
        f"data_parallel_master_ip {parallel_config.data_parallel_master_ip} "
        f"must be the same as launcher host IP {host_ip}"
    )
    _validate_launch_from_rank0_host(mpi_args, host_ip)

    # Remote ranks read the pickled config from a shared directory.
    serialized_config_path = os.path.join(cfg_dir, "tmp_vllm_tt_cfg.pkl")
    with open(serialized_config_path, "wb") as tf:
        cloudpickle.dump(vllm_config, tf)

    # Create a temporary rank binding so vLLM-specific env vars can be injected
    # without mutating the user's source rank-binding file.
    with open(rank_binding_file) as f:
        rb = yaml.safe_load(f)
    rb.setdefault("global_env", {})
    default_env_patterns = ["VLLM_*", "MESH_DEVICE"]
    env_passthrough = tt_config.get("env_passthrough", default_env_patterns)
    if isinstance(env_passthrough, (list, tuple)):
        to_inject = {}
        for key, val in os.environ.items():
            for pattern in env_passthrough:
                if fnmatch.fnmatch(key, pattern):
                    to_inject[key] = val
                    break
        for key, val in to_inject.items():
            rb["global_env"].setdefault(key, val)

    tmp_rb_path = os.path.join(cfg_dir, "tmp_vllm_tt_rank_binding.yaml")
    with open(tmp_rb_path, "w") as tf:
        yaml.safe_dump(rb, tf)

    normalized_extra_ttrun_args: list[str] = []
    if extra_ttrun_args:
        if not isinstance(extra_ttrun_args, str):
            raise RuntimeError(
                "TT plugin config key 'extra_ttrun_args' must be a string"
            )
        normalized_extra_ttrun_args = shlex.split(extra_ttrun_args)
        # These flags are owned by this launcher so it can control config
        # staging and avoid conflicting rank-binding or MPI argument sources.
        reserved_flags = {"--rank-binding", "--mpi-args"}
        if any(
            (tok in reserved_flags)
            or any(tok.startswith(f"{flag}=") for flag in reserved_flags)
            for tok in normalized_extra_ttrun_args
        ):
            raise RuntimeError(
                "TT plugin config key 'extra_ttrun_args' must not include "
                "--rank-binding or --mpi-args"
            )

    # tt-run launches one Python engine entrypoint per MPI rank.
    cmd = ["tt-run"]
    cmd.extend(normalized_extra_ttrun_args)
    cmd.extend(["--rank-binding", tmp_rb_path])
    if mpi_args:
        cmd.extend(["--mpi-args", mpi_args])
    cmd.extend(
        [
            sys.executable,
            "-m",
            "vllm_tt_plugin.launcher",
            "--handshake",
            str(handshake_address),
            "--config-pkl",
            str(serialized_config_path),
            "--log-stats",
            ("1" if log_stats else "0"),
        ]
    )

    logger.info("Launching engines with tt-run: %s", shlex.join(cmd))
    mpi_proc = subprocess.Popen(cmd, env=os.environ.copy())
    _setup_mpi_proc_finalizer(mpi_proc, cleanup_target)


def _setup_mpi_proc_finalizer(
    mpi_proc: subprocess.Popen, cleanup_target: object
) -> None:
    def _finalize_mpi(proc_ref):
        proc = proc_ref()
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            logger.warning(
                "tt-run subprocess did not exit within timeout, sending SIGKILL"
            )
            if proc.pid is not None:
                kill_process_tree(proc.pid)

    weakref.finalize(cleanup_target, _finalize_mpi, weakref.ref(mpi_proc))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TT engine core entrypoint")
    parser.add_argument("--handshake", required=True, help="Handshake address")
    parser.add_argument("--config-pkl", required=True, dest="config_pkl")
    parser.add_argument("--log-stats", required=True, choices=["0", "1"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not os.path.isfile(args.config_pkl):
        raise RuntimeError(f"Config file is not a file: {args.config_pkl}")
    if os.path.islink(args.config_pkl):
        raise RuntimeError(f"Config file is a symlink: {args.config_pkl}")

    with open(args.config_pkl, "rb") as f:
        vllm_config: VllmConfig = cloudpickle.load(f)

    has_mpi = "OMPI_COMM_WORLD_SIZE" in os.environ or "PMI_SIZE" in os.environ
    mpi_rank = int(
        os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "0"))
    )
    mpi_world = int(
        os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "1"))
    )
    if not has_mpi:
        raise RuntimeError("TT engine core must be launched under MPI")

    pc = vllm_config.parallel_config
    assert pc.data_parallel_size % mpi_world == 0
    segment = pc.data_parallel_size // mpi_world
    pc.data_parallel_rank = mpi_rank * segment
    pc.data_parallel_rank_local = 0
    assert pc.distributed_executor_backend == "uni", (
        "TT MPI must be used with uniproc executor backend"
    )

    engine_core_proc_cls = resolve_obj_by_qualname(pc.engine_core_proc_cls)
    engine_core_proc_cls.run_engine_core(
        vllm_config=vllm_config,
        local_client=False,
        handshake_address=args.handshake,
        executor_class=UniProcExecutor,
        log_stats=args.log_stats == "1",
        dp_rank=pc.data_parallel_rank,
        local_dp_rank=pc.data_parallel_rank_local,
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        logger.exception("TT engine core failed")
        sys.exit(1)
