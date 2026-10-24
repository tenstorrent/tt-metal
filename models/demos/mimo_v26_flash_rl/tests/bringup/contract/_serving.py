# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of mimo_v26_flash_rl's serving-contract tests (SC.1, serving_contract.md in the bringup dir).

The served geometry (from the spec and the owner's serving answers), the per-layer KV shapes (from the HF config),
the KV table's config naming the contract fixes, and tt-d-gen's own KV checks loaded from the pinned checkout.
Test harness, not model code: it may read the environment and the framework; the model package never imports it.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

RECORD = 32  # tokens per KV table record (tt-d-gen tools/launch_harness/tables.py:37)
BLOCK = 64  # kv_block_size of every shipped 5120-chunk prefill config (tt-d-gen models/*/dynamo.disagg.prefill.json)
TILE_BYTES = {"bfloat8_b": 1088, "bfloat16": 2048}  # one 32x32 tile (kv_dump_compare.py:40)
KV_DTYPES = {"bfp8": "bfloat8_b", "bfloat8_b": "bfloat8_b", "bf16": "bfloat16", "bfloat16": "bfloat16"}
CONFIG_RE = re.compile(r"(key|value)_h(\d+)")  # one KV table config per (state tensor, KV head)


def hf_config(s) -> dict:
    return json.loads((Path(s.get("paths.hf")) / "config.json").read_text())


def is_swa(cfg: dict, layer: int) -> bool:
    return bool(cfg["hybrid_layer_pattern"][layer])


def kv_heads(cfg: dict, layer: int) -> int:
    return cfg["swa_num_key_value_heads"] if is_swa(cfg, layer) else cfg["num_key_value_heads"]


def widths(cfg: dict, layer: int) -> dict:
    """{state name: row width} of one layer's cache (key after RoPE, value after the value scale)."""
    if is_swa(cfg, layer):
        return {"key": cfg["swa_head_dim"], "value": cfg["swa_v_head_dim"]}
    return {"key": cfg["head_dim"], "value": cfg["v_head_dim"]}


def config_name(name: str, head: int) -> str:
    return f"{name}_h{head}"


def expected_configs(cfg: dict, layers) -> dict:
    """{config name: [layers it covers]}: key_h<n> and value_h<n> for every KV head any served layer has."""
    out = {}
    for name in ("key", "value"):
        for h in range(max(kv_heads(cfg, i) for i in layers)):
            out[config_name(name, h)] = [i for i in layers if h < kv_heads(cfg, i)]
    return out


def slots(s) -> int:
    from bringup_fw.testing import serving as SV

    return int(SV.answer(s, "slots") or s.get("serving.default_answers.slots"))


def kv_dtype(s) -> str:
    """The served KV dtype as a ttnn name (the owner's kv_dtype answer, else the spec's dataformats.kv_cache)."""
    from bringup_fw.testing import serving as SV

    v = SV.answer(s, "kv_dtype") or s.get("dataformats.kv_cache.default") or "bfloat8_b"
    return KV_DTYPES[str(v).lower()]


def geometry(s) -> dict:
    return {
        "chunk": int(s.get("target.chunk")),
        "max_seq": int(s.get("target.seq")),
        "sp": int(s.mesh[0]),
        "tp": int(s.mesh[1]),
        "layers": list(s.layers()),
        "slots": slots(s),
    }


def fabric_mode(s) -> str:
    """PREFILL_FABRIC_MODE for the runner: the fabric the spec's box opens (B.1), so the runner runs the model on the
    fabric the ladder measured (runner_utils.py:41-56 defaults to 2D torus otherwise)."""
    fc = str(s.get("box.device_params", {}).get("fabric_config", ""))
    return {
        "FABRIC_1D": "1d",
        "FABRIC_1D_RING": "1d_ring",
        "FABRIC_2D": "2d",
        "FABRIC_2D_TORUS_X": "2d_torus_x",
        "FABRIC_2D_TORUS_Y": "2d_torus_y",
        "FABRIC_2D_TORUS_XY": "2d_torus_xy",
    }.get(fc, "")


def adapter_name(s) -> str:
    return s.get("contract.adapter") or s.model


# ---------------------------------------------------------------- tt-d-gen, the pinned checkout
def dgen_repo(s) -> Path:
    from bringup_fw.testing import dgen_engine as DE

    return DE.server_repo(s)


def dgen_checks(repo: Path):
    """(tables, kv_dump_compare) from the tt-d-gen checkout: tools/launch_harness/tables.py (read_table: the table
    geometry the server accepts) imported as its package, and kv_manager/tools/kv_dump_compare.py (the server's
    record decoder and PCC). server_rules.harness() cannot load these at this pin: kv_manager/tests/harness/tables.py
    uses relative imports."""
    repo = Path(repo)
    if str(repo / "tools") not in sys.path:
        sys.path.insert(0, str(repo / "tools"))
    import launch_harness.tables as tables

    path = repo / "kv_manager/tools/kv_dump_compare.py"
    spec = importlib.util.spec_from_file_location("dgen_kv_dump_compare", path)
    kvdc = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kvdc
    spec.loader.exec_module(kvdc)
    return tables, kvdc


def transport_problems(repo: Path, workspace: Path, chunk: int, sp: int) -> list[str]:
    """What tt-d-gen's engine and tt-metal's prefill runner must share, checked in their sources: the H2D stream
    service (payload chunk * 4 bytes as [sp, 1, chunk / sp] uint32, 64-byte pages), the 12-byte header
    {slot, start, end} and the layer-ack channel (name and transport). [] when they agree."""
    repo, workspace = Path(repo), Path(workspace)
    out = []
    types = (repo / "engine/include/engine/pipeline/pipeline_types.hpp").read_text()
    if "sizeof(PrefillMetadata) == 12" not in types:
        out.append("tt-d-gen's PrefillMetadata is no longer 12 bytes (pipeline_types.hpp)")
    runner = (workspace / "models/demos/common/prefill/runners/prefill_runner.py").read_text()
    if not re.search(r"^METADATA_SIZE_BYTES = 12$", runner, re.M):
        out.append("tt-metal's prefill_runner.py METADATA_SIZE_BYTES is not 12")
    if chunk % sp or (chunk // sp * 4) % 64:
        out.append(
            f"chunk {chunk} over sp {sp}: the per-chip H2D page ({chunk // sp * 4} B) must be a whole number "
            "of 64-byte pages (runner_utils.py:133-138)"
        )
    counter = repo / "engine/include/engine/pipeline/inter_process_counter_channel.hpp"
    ring = (workspace / "models/demos/common/prefill/runners/layer_completion_ring_reader.py").read_text()
    m = re.search(r'return f"(/tt_prefill_layer_\w+_)\{service_id\}"', ring)
    runner_ack = m.group(1) if m else "?"
    if counter.exists():
        out.append(
            f"layer acks: the pinned tt-d-gen attaches an InterProcessCounterChannel at "
            f"/tt_prefill_layer_acks_<service_id>; tt-metal's runner publishes a LayerCompletionQueue ring at "
            f"{runner_ack}<service_id> (layer_completion_ring_reader.py) and has no counter channel. tt-d-gen "
            "#1600 (1cf34388) moves the engine to the ring: pin tt-d-gen at or after it"
        )
        return out
    m = re.search(r'ack = "(/tt_prefill_layer_\w+_)" \+ service', (repo / "tools/launch_harness/config.py").read_text())
    engine_ack = m.group(1) if m else "?"
    if engine_ack != runner_ack:
        out.append(
            f"layer acks: tt-d-gen attaches {engine_ack}<service_id> (tools/launch_harness/config.py), tt-metal's "
            f"runner publishes {runner_ack}<service_id> (layer_completion_ring_reader.py scheduler_shm_name)"
        )
    # The ring's wire format: tt-d-gen links the LayerCompletionQueue of its tt-blaze tt-metal, the runner the
    # workspace's. Same message struct and queue header (comments aside), or the engine reads garbage.
    blaze = repo / "third_party/tt-blaze/tt-metal"
    for rel in (
        "tt_metal/api/internal/disaggregation/layer_completion_message.hpp",
        "tt_metal/api/internal/disaggregation/layer_completion_queue.hpp",
    ):
        a, b = workspace / rel, blaze / rel
        if not a.is_file() or not b.is_file():
            out.append(f"layer acks: {rel} missing in {'the workspace' if not a.is_file() else blaze}")
        elif _code_lines(a) != _code_lines(b):
            out.append(
                f"layer acks: {rel} differs between tt-d-gen's tt-blaze tt-metal and the workspace (ring wire format)"
            )
    return out


def _code_lines(p: Path) -> list[str]:
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.strip().startswith("//")]


def driver_shell(build, plan: str, out: str, rc_file: str, service_id: str) -> list[str]:
    """dgen_engine.driver_shell, with the driver started through _dgen_driver_shim.py (the pinned engine's
    device_prefill_pipeline takes no layers_per_chunk: the ring header carries completions_per_chunk)."""
    import shlex

    from bringup_fw.testing import dgen_engine as DE

    shim = str(Path(__file__).resolve().with_name("_dgen_driver_shim.py"))
    cmd = DE.driver_shell(build, plan, out, rc_file, service_id)
    old = f"{shlex.quote(build.python)} {shlex.quote(str(DE.DRIVER))}"
    if old not in cmd[-1]:
        raise RuntimeError(f"dgen_engine.driver_shell changed shape; cannot insert the shim: {cmd[-1][:300]}")
    cmd[-1] = cmd[-1].replace(old, f"{shlex.quote(build.python)} {shlex.quote(shim)} {shlex.quote(str(DE.DRIVER))}", 1)
    return cmd
