# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the gemma4_31b_glx serving-contract tests (serving_contract.md).

Imports nothing heavy at module level: the tests must collect under BRINGUP_IMPL=reference on a host without a mesh.

The served KV layout (serving_contract.md, "KV cache" / "Adapter and table"):
  36 table configs, in this order (the decode side reads them by name):
    00..03  NN_global_h<i>     global layers (layer % 6 == 5), head i, packed 640-wide row
                               [K rotary (128, adjacent pairs) | V non-rotary (384) | V rotary (128)]
    04..19  NN_sliding_k_h<i>  sliding layers, K of head i, 256 wide, channels in adjacent-pair order
    20..35  NN_sliding_v_h<i>  sliding layers, V of head i, 256 wide, HF channel order
  one table record = 32 tokens of one (slot, layer, config) in one DRAM bank.
"""

from __future__ import annotations

import os

GLOBAL_HEADS, SLIDING_HEADS = 4, 16
GLOBAL_DIM, GLOBAL_ROTARY, GLOBAL_PACKED, SLIDING_DIM = 512, 128, 640, 256
CONFIG_NAMES = (
    tuple(f"{i:02d}_global_h{i}" for i in range(GLOBAL_HEADS))
    + tuple(f"{i + 4:02d}_sliding_k_h{i}" for i in range(SLIDING_HEADS))
    + tuple(f"{i + 20:02d}_sliding_v_h{i}" for i in range(SLIDING_HEADS))
)
GLOBAL_CONFIGS = tuple(range(0, 4))
SLIDING_CONFIGS = tuple(range(4, 36))
# A sample of configs per layer type that the KV read-back checks (first and last head of each tensor): reading every
# record of every config through UMD would move tens of GB.
SAMPLE_CONFIGS = {"global": (0, 3), "sliding": (4, 19, 20, 35)}
KV_DTYPES = {"bfp8": "bfloat8_b", "bfloat8_b": "bfloat8_b", "bf16": "bfloat16", "bfloat16": "bfloat16"}
DUMP_DTYPE = {"bfloat8_b": "bfp8", "bfloat16": "bf16"}


def is_global(layer: int) -> bool:
    """Gemma 4 31B layer_types: every sixth layer (5, 11, ..., 59) is full attention."""
    return layer % 6 == 5


def layer_configs(layer: int) -> tuple[int, ...]:
    return GLOBAL_CONFIGS if is_global(layer) else SLIDING_CONFIGS


def config_width(config: int) -> int:
    return GLOBAL_PACKED if config in GLOBAL_CONFIGS else SLIDING_DIM


def not_built(msg: str):
    import pytest

    pytest.fail(f"not built: {msg}", pytrace=False)


def kv_dtype_name(s) -> str:
    """The served KV dtype as a ttnn dtype name (the owner's kv_dtype answer, default bfp8)."""
    from bringup_fw.testing import serving

    v = str(serving.answer(s, "kv_dtype") or "bfp8").lower()
    return KV_DTYPES.get(v, v)


def slots(s) -> int:
    from bringup_fw.testing import serving

    return int(serving.answer(s, "slots") or s.get("contract.num_users") or 2)


def geometry(s) -> tuple[int, int, int]:
    """(chunk, max_seq, sp) the server runs: target chunk / seq (the all-chunks target rung)."""
    return int(s.get("target.chunk")), int(s.get("target.seq")), int(s.mesh[0])


def target_golden(s):
    from bringup_fw.reference.golden import Golden
    from bringup_fw.testing.chunk_starts import target_rung

    rung = target_rung(s)
    if rung is None:
        not_built("no all-chunks rung at the target length in the spec's ladder")
    return Golden.for_rung(s, rung["name"])


# ---------------------------------------------------------------- golden KV in the served layout
def _pairs(lo_a: int, lo_b: int, n: int):
    import torch

    return torch.stack((torch.arange(lo_a, lo_a + n), torch.arange(lo_b, lo_b + n)), dim=1).reshape(-1)


def global_pack_indices():
    """(K rotary columns, V columns) of the packed global row, from HF's NeoX head layout."""
    import torch

    half, rh = GLOBAL_DIM // 2, GLOBAL_ROTARY // 2
    rotary = _pairs(0, half, rh)
    nonrotary = torch.cat((torch.arange(rh, half), torch.arange(half + rh, GLOBAL_DIM)))
    value = torch.cat((nonrotary, torch.arange(0, rh), torch.arange(half, half + rh)))
    return rotary, value


def golden_config_rows(state: dict, layer: int, config: int, n: int):
    """Rows [0, n) of one table config, from the golden / reference state {"key", "value"} [heads, seq, dim]."""
    key, value = state["key"].float(), state["value"].float()
    if key.dim() == 4:  # [1, heads, seq, dim]
        key, value = key[0], value[0]
    key, value = key[:, :n], value[:, :n]
    if is_global(layer):
        rotary, vcols = global_pack_indices()
        import torch

        h = config - GLOBAL_CONFIGS[0]
        return torch.cat((key[h].index_select(-1, rotary), value[h].index_select(-1, vcols)), dim=-1)
    if config < 20:
        return key[config - 4].index_select(-1, _pairs(0, SLIDING_DIM // 2, SLIDING_DIM // 2))
    return value[config - 20]


# ---------------------------------------------------------------- tt-d-gen's own checks
def dgen_checks():
    """(tables, kv_dump_compare) from the pinned tt-d-gen checkout. server_rules.harness() copies tables.py alone and
    its relative import (`from .config import ...`) fails at the pin, so load the harness as a package then."""
    import importlib
    import importlib.util
    import sys
    from pathlib import Path

    from bringup_fw.testing import dgen_engine
    from bringup_fw.testing import server_rules as R

    repo = Path(dgen_engine.server_repo())
    try:
        return R.harness(repo)
    except ImportError:
        pass
    hdir = repo / "kv_manager/tests/harness"
    if "dgen_kvh" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "dgen_kvh", hdir / "__init__.py", submodule_search_locations=[str(hdir)]
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["dgen_kvh"] = mod
        spec.loader.exec_module(mod)
    tables = importlib.import_module("dgen_kvh.tables")
    spec = importlib.util.spec_from_file_location("dgen_kv_dump_compare", repo / "kv_manager/tools/kv_dump_compare.py")
    kvc = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kvc
    spec.loader.exec_module(kvc)
    return tables, kvc


def table_failures(table, num_layers: int, n_slots: int, max_seq: int, tables_mod, dtype: str = "bfp8") -> list[str]:
    """The exported table against the server's rules: config names and geometry (tt-d-gen tables.py table_profile),
    per layer type full coverage and kv_table.layout_failures (migration_strategy_builder.cpp: a config covers only its
    own layer type; every covered record present, one size per config, aligned, no aliasing or overlap)."""
    from types import SimpleNamespace

    from bringup_fw.testing import kv_table

    fails = []
    names = tuple(table.config_name(i) for i in range(table.num_configs()))
    if names != CONFIG_NAMES:
        fails.append(f"table configs {names[:5]}... != the served order {CONFIG_NAMES[:5]}... ({len(names)} vs 36)")
        return fails
    for c in range(table.num_configs()):
        cfg = table.config(c)
        engine = SimpleNamespace(
            num_layers=num_layers,
            max_seq_tokens=max_seq,
            chunk_tokens=32,
            chunk_size_bytes=int(cfg.chunk_size_bytes),
            dram_base=None,
        )
        ep = SimpleNamespace(id=0, name="prefill-0", num_slots=n_slots, hosts=["prefill-0"], table_spec={})
        prof = tables_mod.table_profile(SimpleNamespace(engine=engine, epochs=()), ep)
        got = {
            "layers": int(cfg.num_layers),
            "max_seq_len": int(cfg.max_sequence_length),
            "slots": int(cfg.num_slots),
            "chunk_n_tokens": int(cfg.chunk_n_tokens),
        }
        bad = {k: (v, prof[k]) for k, v in got.items() if v != prof[k]}
        if bad:
            fails.append(f"config {names[c]}: geometry {bad} (table, tables.py profile)")
        want_bytes = config_width(c) // 32 * (1088 if dtype == "bfp8" else 2048)  # one 32-row tile per 32 columns
        if int(cfg.chunk_size_bytes) != want_bytes:
            fails.append(
                f"config {names[c]}: chunk_size_bytes {cfg.chunk_size_bytes} != {want_bytes} "
                f"(32 x {config_width(c)} tiles)"
            )
    for kind, configs in (("global", GLOBAL_CONFIGS), ("sliding", SLIDING_CONFIGS)):
        layers = [i for i in range(num_layers) if (kind == "global") == is_global(i)]
        records = {}
        for k, c in enumerate(configs):
            for s in range(n_slots):
                for layer in layers:
                    for p in range(0, max_seq, 32):
                        loc = table.lookup(layer, p, s, c)
                        records[(s, layer, p, k)] = (
                            int(loc.noc_addr),
                            int(loc.size_bytes),
                            int(loc.device_group_index),
                        )
        zero = [k for k, v in records.items() if v[1] == 0]
        if zero:
            fails.append(f"{kind}: {len(zero)} records of its own layers unset, e.g. {zero[:3]}")
        fails += [f"{kind}: {f}" for f in kv_table.layout_failures(records, layers, n_slots, max_seq, len(configs))]
        other = [i for i in range(num_layers) if i not in layers][:1]
        if other and int(table.lookup(other[0], 0, 0, configs[0]).size_bytes):
            fails.append(f"{kind} config {configs[0]} also maps layer {other[0]} of the other type")
    return fails


def read_back_failures(
    reader, kvc, golden_state, slot: int, n: int, layers, thr: float, dtype: str, tag: str, configs_of=None
) -> tuple[list[str], dict]:
    """KV rows [0, n) of ``slot`` read through the table, decoded and scored with tt-d-gen's kv_dump_compare
    (decode_chunk / reassemble / pcc) against the golden mapped into the served layout; the pad rows of the last
    record must be zero. Returns (fails, {f"{layer}:{config}": pcc})."""
    import numpy as np

    fails, scores = [], {}
    end = -(-n // 32) * 32
    for layer in layers:
        st = golden_state(layer)
        for c in (configs_of or (lambda i: SAMPLE_CONFIGS["global" if is_global(i) else "sliding"]))(layer):
            w = config_width(c)
            dump = {(layer, p): reader.record(slot, layer, p, c) for p in range(0, end, 32)}
            rows = kvc.reassemble(dump, layer, 0, end, dtype=dtype, width=w, storage="tile")
            want = golden_config_rows(st, layer, c, n).numpy()
            p = kvc.pcc(rows[:n], want)
            scores[f"{layer}:{c}"] = p
            if not (p == p and p >= thr):
                fails.append(f"{tag}: slot {slot} layer {layer} {CONFIG_NAMES[c]} rows [0, {n}) pcc {p:.5f} < {thr}")
            pad = rows[n:end]
            if pad.size and np.any(pad != 0):
                fails.append(f"{tag}: slot {slot} layer {layer} {CONFIG_NAMES[c]} pad rows [{n}, {end}) not zero")
    if reader.replica_mismatch:
        fails.append(
            f"{tag}: {len(reader.replica_mismatch)} records differ between the chips of their device group, "
            f"e.g. {reader.replica_mismatch[:3]}"
        )
    return fails, scores


def sample_layers(num_layers: int) -> list[int]:
    """First and last sliding and global layers: what the read-back samples."""
    g = [i for i in range(num_layers) if is_global(i)]
    sl = [i for i in range(num_layers) if not is_global(i)]
    return sorted({sl[0], sl[-1], g[0], g[-1]} if g else {sl[0], sl[-1]})


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"


# ---------------------------------------------------------------- the adapter (K.1) in this process
def adapter_runtime(s, mesh):
    """(adapter, kv, runtime) for the served geometry through tt-metal's prefill adapter API, as prefill_runner.py
    builds them; "not built" while the adapter, its runtime or a runtime method is missing."""
    from bringup_fw.testing import contract, kv_table

    env = contract.engine_env(s)
    try:
        from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter

        adapter = get_adapter(env["PREFILL_MODEL"])
    except (KeyError, ImportError, NotImplementedError) as e:
        not_built(f"prefill adapter {env['PREFILL_MODEL']!r} ({type(e).__name__}: {e}); register it in ADAPTER_PATHS")
    chunk, max_seq, _ = geometry(s)
    first, n_layers = contract.served_layers(s)
    hf = adapter.load_hf_config()
    params = PrefillRunParams(
        mesh_shape=tuple(s.mesh),
        num_layers=n_layers,
        first_layer_idx=first,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=max_seq,
        chunk_size=chunk,
        num_users=slots(s),
        capacity_factor=1,
        num_links=int(s.get("contract.num_links")),
        gate_mode_name=adapter.default_gate_mode,
        kv_only_last_layer=False,
        weight_cache_path=adapter.weight_cache_path(tuple(s.mesh)),
    )
    kv = adapter.allocate_kv_cache(mesh_device=mesh, hf_config=hf, params=params)
    rt = adapter.build_runtime(mesh_device=mesh, hf_config=hf, params=params)
    missing = kv_table.check_runtime(rt)
    if missing:
        not_built("the runtime lacks " + "; ".join(missing))
    return adapter, kv, rt


def export_table(rt, kv, out_dir):
    import ttnn

    path = os.path.join(str(out_dir), "kv_table.pb")
    path = rt.build_kv_chunk_table(kv, path) or path
    return ttnn.experimental.disaggregation.import_from_protobuf_file(path)


def server_input(mesh, chunk_tokens, start: int, end: int, chunk: int, sp: int):
    """One chunk exactly as tt-d-gen's H2D stream delivers it: PAD_ID past actual_end, reshuffled across the SP chips
    by actual_start (server_rules.server_payload), [sp, 1, chunk / sp] uint32 sharded on the SP axis."""
    import torch
    from bringup_fw.testing import server_rules as R

    import ttnn

    payload = torch.from_numpy(R.server_payload(chunk_tokens, start, end, chunk, sp).astype("int64")).to(torch.uint32)
    return ttnn.from_torch(
        payload,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh, ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()])
        ),
    )
