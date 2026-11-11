# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the deepseek_v3_d_p prefill stack (TtMoe): DeepSeek-V3, GLM-5.3, Kimi-K2.7, Kimi-K3, Mistral-Small-4.

All four hooks already exist in this stack:
  * routing capture   TtMoe's PREFILL_DUMP_ROUTING_DIR (untraced)
  * on-device remap   moe_grouped_topk(expert_id_map=...) loaded from the placed cache's per-layer markers
                      (models/demos/deepseek_v3_d_p/tt/moe/expert_placement.py)
  * placed cache      models/demos/deepseek_v3_d_p/scripts/prepare_expert_placement.py (byte-verified shards)
  * timed/golden runs one parametrization of tests/test_prefill_transformer_chunked.py, driven by dsv3dp_bench.py

Paths come from the environment, through the model's runner adapter (models.demos.common.prefill.adapter):
<env_var> = HF model directory (tokenizer, config), <ttnn_cache_env> = parent of <variant>_bh_32dev. A golden
sequence directory can be given with MOE_PLACEMENT_GOLDEN_<MODEL> or in the model entry.

``verified`` marks models with an end-to-end on-device A/B through this pipeline; the others are configured
from their test parametrization and still need a first smoke run (``capture`` on one sequence).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..spec import MoeSpec
from .base import PlacementAdapter

TEST_FILE = "models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py"
ITER_IDS = {"1": "iters1", "2": "two_iters", "10": "ten_iters"}


@dataclass(frozen=True)
class ModelEntry:
    variant: str  # name in the prefill adapter registry and the test's ``variant`` fixture
    num_experts: int
    top_k: int
    transport: str  # cost model matching TtMoe's transport on this mesh: ring (fabric2d on a torus) | linear (direct)
    test: str
    mesh_id: str
    num_layers: int  # depth of the timed runs (deepest the test offers / one Galaxy holds)
    n_chunks: int = 11
    chunks_id: str = "chunks_eleven"
    extra_selector: tuple = ("preload0",)
    run_overrides: dict = field(default_factory=dict)  # "@NAME" = attribute of the test module
    perf_overrides: dict = field(default_factory=dict)
    chat_format: str = "hf"
    golden: str | None = None
    verified: bool = False


MODELS = {
    "glm_5_3": ModelEntry(
        variant="glm_5_3",
        num_experts=256,
        top_k=8,
        transport="ring",
        test="test_glm_prefill_transformer_chunked_no_pcc",
        mesh_id="torus-xy-8x4",
        num_layers=78,
        run_overrides={"seq_cache": "@SEQ_CACHE"},
        perf_overrides={"kv_pcc_threshold": "@KV_CACHE_PCC_THRESHOLD_SCALED_FP8"},
        chat_format="glm",
        golden="/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/golden_traces/vllm-glm53-indexer-kcache-55k",
        verified=True,
    ),
    "kimi_k3": ModelEntry(
        variant="kimi_k3",
        num_experts=896,
        top_k=16,
        transport="linear",
        test="test_kimi_k3_prefill_transformer_chunked",
        mesh_id="fabric2d-8x4",
        num_layers=24,
        extra_selector=("preload0", "no_determinism"),
        golden="/mnt/weka/model-cache/stable/deepseek-prefill-cache/golden/k3_vllm_code_debug_1M",
        verified=True,
    ),
    "kimi_k2_7": ModelEntry(
        variant="kimi_k2_7",
        num_experts=384,
        top_k=8,
        transport="ring",
        test="test_kimi_prefill_transformer_chunked",
        mesh_id="torus-xy-8x4",
        num_layers=61,
        extra_selector=("preload0", "margin_auto"),
    ),
    "deepseek_v3": ModelEntry(
        variant="deepseek_v3_d_p",
        num_experts=256,
        top_k=8,
        transport="ring",
        test="test_ds_prefill_transformer_chunked_no_pcc",
        mesh_id="torus-xy-8x4",
        num_layers=61,
    ),
    "mistral_small_4": ModelEntry(
        variant="mistral_small_4",
        num_experts=128,
        top_k=4,
        transport="ring",
        test="test_mistral4_prefill_transformer_chunked_no_pcc",
        mesh_id="torus-xy-8x4",
        num_layers=36,
        n_chunks=10,
        chunks_id="chunks10",
        extra_selector=(),
    ),
}

TRANSPORT_NAMES = {"ring": "fabric2d", "linear": "direct"}  # cost model -> TtMoe.moe_transport


def _raise_nproc():
    """32-chip runs start thousands of threads; low default per-user process limits make them fail late."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != resource.RLIM_INFINITY and soft < 8192:
        resource.setrlimit(resource.RLIMIT_NPROC, (min(8192, hard) if hard != resource.RLIM_INFINITY else 8192, hard))


class DeepseekV3DPAdapter(PlacementAdapter):
    family = "deepseek_v3_d_p"

    def __init__(self, model: str, repo: Path | None = None):
        if model not in MODELS:
            raise KeyError(f"unknown deepseek_v3_d_p model {model!r}; known: {sorted(MODELS)}")
        self.model = model
        self.entry = e = MODELS[model]
        self.verified = e.verified
        spec = MoeSpec(
            name=model,
            num_experts=e.num_experts,
            top_k=e.top_k,
            rows=8,
            groups=4,
            transport=e.transport,
            n_chunks=e.n_chunks,
        )
        super().__init__(spec, e.chat_format)
        self.repo = Path(repo or os.environ.get("TT_METAL_HOME", Path(__file__).resolve().parents[4]))

    # --- paths -----------------------------------------------------------------------------------
    def _runner_adapter(self):
        from models.demos.common.prefill.adapter import get_adapter

        return get_adapter(self.entry.variant)

    def _env_names(self) -> tuple[str, str]:
        a = self._runner_adapter()
        return a.env_var, a.ttnn_cache_env

    def tokenizer_path(self) -> str:
        env, _ = self._env_names()
        if not os.environ.get(env):
            raise RuntimeError(f"set {env} to the {self.model} HF model directory")
        return os.environ[env]

    def original_cache(self) -> Path:
        _, env = self._env_names()
        if not os.environ.get(env):
            raise RuntimeError(f"set {env} to the parent of {self.entry.variant}_bh_32dev")
        return Path(os.environ[env]) / f"{self.entry.variant}_bh_32dev"

    def golden_sequence(self) -> Path | None:
        g = os.environ.get(f"MOE_PLACEMENT_GOLDEN_{self.model.upper()}", self.entry.golden)
        if not g:
            return None
        g = Path(g)
        if (g / "metadata.json").exists():
            return g
        if g.is_dir():  # some golden traces nest one level down
            return next((d for d in sorted(g.iterdir()) if (d / "metadata.json").exists()), None)
        return None

    # --- device stages -------------------------------------------------------------------------
    def _bench(
        self, mode: str, sequence: Path, cache: Path, out_dir: Path, log: Path, num_iters: int = 10, timeout: int = 3600
    ) -> int:
        e = self.entry
        cfg = dict(
            mode=mode,
            model=self.model,
            variant=e.variant,
            output=str(out_dir),
            cache=str(cache),
            num_layers=e.num_layers,
            n_chunks=e.n_chunks,
            chunk_tokens=self.spec.chunk_tokens,
            num_iters=num_iters,
            num_experts=e.num_experts,
            top_k=e.top_k,
            test_file=TEST_FILE,
            test=e.test,
            mesh_id=e.mesh_id,
            chunks_id=e.chunks_id,
            extra_selector=list(e.extra_selector),
            iter_ids=ITER_IDS,
            run_overrides=e.run_overrides,
            perf_overrides=e.perf_overrides,
        )
        log = Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        cfg_path = log.with_suffix(".config.json")
        cfg_path.write_text(json.dumps(cfg, indent=1))
        env = dict(
            os.environ,
            TT_METAL_HOME=str(self.repo),
            PREFILL_TRACE_DIR=str(sequence),
            LOGURU_LEVEL="INFO",
            TT_METAL_DEVICE_PROFILER="0",
        )
        env["PYTHONPATH"] = f"{self.repo}:{self.repo / 'ttnn'}" + (
            ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        # the timed harness checks chip power limits; placement A/Bs compare two caches on one node instead
        for k, v in dict(DS_PERF_IGNORE_POWER="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1").items():
            env.setdefault(k, v)
        for k in (
            "PREFILL_DUMP_ROUTING_DIR",
            "PREFILL_DUMP_ROUTING_LAYERS",
            "TT_PREFILL_BLOCK_TIMING",
            "TT_MESH_GRAPH_DESC_PATH",
        ):
            env.pop(k, None)
        with log.open("w") as f:
            return subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "models.common.moe_placement.adapters.dsv3dp_bench",
                    "--config",
                    str(cfg_path),
                ],
                cwd=self.repo,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                preexec_fn=_raise_nproc,
            ).returncode

    def _check_transport(self, out_dir: Path) -> None:
        got = json.loads((out_dir / "measurement.json").read_text()).get("moe_transport") or []
        want = TRANSPORT_NAMES[self.spec.transport]
        if got and got != [want]:
            raise RuntimeError(
                f"TtMoe ran transport {got}, but the {self.model} entry models {want!r}: fix ModelEntry.transport before calibrating"
            )

    def capture(self, sequence: Path, out_dir: Path, log: Path) -> dict:
        rc = self._bench("capture", sequence, self.original_cache(), out_dir, log)
        m = json.loads((out_dir / "measurement.json").read_text()) if (out_dir / "measurement.json").exists() else {}
        if rc != 0 or m.get("status") != "complete":
            raise RuntimeError(f"capture of {sequence.name} failed (rc {rc}), see {log}")
        self._check_transport(out_dir)
        return m

    def run(self, mode: str, sequence: Path, cache: Path, out_dir: Path, log: Path) -> int:
        rc = self._bench(mode, sequence, cache, out_dir, log)
        if rc == 0:
            self._check_transport(out_dir)
        return rc

    def prepare_cache(self, preset: Path, destination: Path, log: Path) -> None:
        dest = Path(destination)
        env = dict(os.environ, TT_METAL_HOME=str(self.repo), PYTHONPATH=f"{self.repo}:{self.repo / 'ttnn'}")
        with Path(log).open("w") as f:
            rc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "models.demos.deepseek_v3_d_p.scripts.prepare_expert_placement",
                    "--source",
                    str(self.original_cache() / "8x4"),
                    "--destination",
                    str(dest / "8x4"),
                    "--plan",
                    str(preset),
                ],
                cwd=self.repo,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=6 * 3600,
            ).returncode
        if rc != 0:
            raise RuntimeError(f"cache preparation failed (rc {rc}), see {log}")
