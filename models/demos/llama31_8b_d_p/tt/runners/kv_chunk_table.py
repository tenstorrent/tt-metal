# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The KV-chunk **address table**: `(layer, position, slot, config)` -> a DRAM address, for migration.

**HF anchor:** none — this describes `tt/attention/kv_cache.py`'s DRAM layout, not model math.
**Template:** `models/demos/gpt_oss_d_p/tt/runners/kv_chunk_table.py:66`
(`build_kv_chunk_address_table`) and `:179` (`build_and_serialize_kv_chunk_table`).

**The template's builder is IMPORTED, not copied** (`DEC-099`). Agent-contract rule 4 says reuse
means import, and this is the case it was written for: P5.6 deliberately kept this package's cache
byte-for-byte structural with gpt-oss's — the same `NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32`, the
same `[1, 1, 32, head_dim]` `NdShardSpec`, the same `ROUND_ROBIN_1D` distribution, the same
user-major `slot = user * num_layers + layer` packing — precisely so that P10 could reuse the
address walk instead of writing a second one (`tt/attention/kv_cache.py:17-30`,
`bringup_log/03_OUTLINE.md` §2.7). `head_dim` and `num_kv_heads` are already parameters there, and
the two values this model differs in (128 and 8) go in through them.

What the import buys, and what it costs:

* **buys** — one bank walk, in one place, for the two packages whose caches are the same shape. A
  copy would be 140 lines of address arithmetic that `G-KV-TABLE` would gate here and
  `models/demos/gpt_oss_d_p/tests/test_kv_cache_table.py:256` would gate there, drifting silently
  in between.
* **costs** — a cross-package dependency on a *runners* module, and an upstream change to that
  bank walk would land here unannounced. Two guards: `_assert_layout_still_shared()` fails loudly
  if the two packages' block constant ever diverges, and `G-KV-TABLE` reads the resulting addresses
  back over UMD and compares **bit-exactly** against the live cache, so a drifted walk fails as a
  wrong address rather than as a wrong PCC (recipe §2.5).

**What is NOT implemented, and raises.** The **multi-rank merge**: with pipeline parallelism each
rank owns a layer slice and the merged table has to be assembled from every rank's gathered stage
layout (`models/demos/common/prefill/runners/migration.py:287` `allgather_kv_stage_layouts`). The
engine passes `first_layer_idx`, `num_my_layers` and `stage_layout` on three of its five call sites
(`models/demos/common/prefill/runners/prefill_runner.py:644`, `:655`, `:674`) and the template
**discards** them with a `del` (`models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:388`). Recipe P10
step 4 requires the opposite — "must **raise**, naming its risk id, rather than silently discarding
an argument" — so `assert_single_rank_stage` refuses anything but this rank owning the whole model,
naming `R-032` (multi-galaxy is out of scope by user instruction).

**Read `assert_single_rank_stage`'s docstring before touching it.** Its first version got the
`stage_layout` type wrong *and* wrote two guards that compared a value with itself, so it would have
rejected every real migration run while providing none of the protection it advertised. `DEC-111`
records both, and the shape of the mistake is `DEC-108`'s: a refusal written from the parameter's
*name* rather than from what the engine actually puts in it.
"""

from __future__ import annotations

from loguru import logger

from models.demos.common.prefill.runners.migration import serialize_prebuilt_kv_chunk_table
from models.demos.gpt_oss_d_p.tt.attention.kv_cache import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK as _TEMPLATE_TOKENS_PER_BANK_BLOCK,
)
from models.demos.gpt_oss_d_p.tt.runners.kv_chunk_table import build_kv_chunk_address_table as _build_shared_table

from ..attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK


def _assert_layout_still_shared() -> None:
    """The one structural fact the imported bank walk assumes about *our* cache.

    `models/demos/gpt_oss_d_p/tt/runners/kv_chunk_table.py:55` writes the template's own constant
    into `chunk_n_tokens`, so if the two packages' block sizes ever diverge the table would describe
    32-token chunks over a cache laid out in some other block size — addresses that resolve, decode,
    and are wrong. Fail at build time instead.
    """
    if NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK != _TEMPLATE_TOKENS_PER_BANK_BLOCK:
        raise AssertionError(
            f"this package's DRAM block is {NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK} tokens but the "
            f"imported gpt-oss table builder assumes {_TEMPLATE_TOKENS_PER_BANK_BLOCK} "
            f"(models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:27). The shared bank walk is only "
            f"valid while the two layouts match (DEC-099); write a local builder, or realign the "
            f"constant in tt/attention/kv_cache.py."
        )


def assert_single_rank_stage(*, num_layers: int, first_layer_idx: int, num_my_layers=None, stage_layout=None) -> None:
    """Refuse a pipeline-parallel table request. See the module docstring; risk `R-032`.

    **`stage_layout` is a LIST of one dict per rank, not a dict** — and getting that wrong is what
    `DEC-111` is about. `allgather_kv_stage_layout` builds it with `for rk in range(size)`
    (`models/demos/common/prefill/runners/migration.py:315-334`), `allgather_kv_stage_layouts`
    returns one such list per migratable stage (`:287-291`), and the engine hands
    `stage_layouts[0]` — i.e. **stage 0's per-rank list** — through as `stage_layout`
    (`prefill_runner.py:634`). Every other reader in the tree iterates it:
    `models/demos/deepseek_v3_d_p/tt/tt_prefill_runtime.py:929` tests single-rank with
    `all(len(layout) == 1 for layout in stage_layouts)` and
    `models/demos/deepseek_v3_d_p/utils/kv_cache_utils.py:394` sums `s["count"]` over it.

    **The gathered list is also the only argument that can actually detect a pipeline rank**, which
    is why the first draft's guards were vacuous (`DEC-111`): the engine builds the table on rank 0
    only (`prefill_runner.py:643`, `:654`, `:673`), so `first_layer_idx` is always 0 there; and
    `num_my_layers` is compared against `config.num_layers`, which **is** this rank's
    `num_my_layers` (`prefill_runner.py:463`, `:481`), so that test compared a value with itself.
    The list's **length** and its summed `count` are the two facts that come from every rank.

    Args:
        num_layers: this rank's layer extent, i.e. `config.num_layers`.
        first_layer_idx: the engine's `first_layer_idx` for this rank. 0 on every path that builds.
        num_my_layers: the engine's per-rank layer count, or `None` when it did not pass one.
        stage_layout: the gathered **list** of per-rank stage dicts, or `None` on the pure-mock path
            that gathers nothing (`prefill_runner.py:570`, `:699`).
    """
    if first_layer_idx != 0:
        raise NotImplementedError(
            f"build_kv_chunk_table was asked for a table starting at layer {first_layer_idx}, i.e. a "
            f"pipeline rank that is not rank 0. Merging per-rank layer slices into one table is "
            f"unimplemented and multi-galaxy pipeline parallelism is out of scope for this iteration "
            f"(BRINGUP_RECIPE.md line 16, risk R-032). Run single-rank."
        )
    if num_my_layers is not None and int(num_my_layers) != int(num_layers):
        raise NotImplementedError(
            f"build_kv_chunk_table was asked for {num_my_layers} layers but this runtime was built "
            f"for {num_layers}. The two are the same number on every engine path "
            f"(prefill_runner.py:481 passes num_my_layers as PrefillRunParams.num_layers), so a "
            f"disagreement means the caller is not the engine (risk R-032)."
        )
    if stage_layout is None:
        return
    if isinstance(stage_layout, dict):
        raise TypeError(
            "stage_layout must be the gathered LIST of per-rank stage dicts "
            "(models/demos/common/prefill/runners/migration.py:315-334), not a single dict. The "
            "engine passes stage_layouts[0], which is stage 0's per-rank list "
            "(prefill_runner.py:634). Passing one dict is DEC-111's bug."
        )
    if not isinstance(stage_layout, (list, tuple)) or not stage_layout:
        raise TypeError(
            f"stage_layout must be a non-empty sequence of per-rank stage dicts, got "
            f"{type(stage_layout).__name__}. A list under the key `stage_layouts` (plural) reaches a "
            f"runtime that declares kv_migration_stages, which this one deliberately does not "
            f"(DEC-107, prefill_runner.py:613)."
        )
    if len(stage_layout) != 1:
        ranks = [(s.get("rank"), s.get("first_layer"), s.get("count")) for s in stage_layout]
        raise NotImplementedError(
            f"the gathered stage layout carries {len(stage_layout)} ranks {ranks}, so this is a "
            f"pipeline-parallel run. Merging per-rank layer slices into one table is unimplemented "
            f"and multi-galaxy is out of scope by user instruction (risk R-032). This builder emits "
            f"a whole-model, single-rank table."
        )
    stage = stage_layout[0]
    total = int(stage.get("count", num_layers))
    if int(stage.get("first_layer", 0)) != 0 or total != int(num_layers):
        raise NotImplementedError(
            f"the gathered stage describes layers "
            f"[{stage.get('first_layer')}, +{stage.get('count')}) of this runtime's {num_layers}, so "
            f"it is not the whole model. Merging is unimplemented (risk R-032)."
        )


def build_kv_chunk_address_table(
    *,
    mesh_device,
    kv_cache,
    seq_len,
    num_layers,
    mesh_shape,
    sp_axis,
    num_users,
    chunk_size,
    num_kv_heads,
    head_dim,
):
    """The multi-config block-cyclic table for `kv_cache` (does **not** serialize).

    `2 * num_kv_heads` configs, in the order the src<->dst migration contract fixes:
    `config h` is K head `h`, `config num_kv_heads + h` is V head `h`
    (`models/demos/gpt_oss_d_p/tt/runners/kv_chunk_table.py:13-18`). Head `h` lives on TP column
    `h` — the equality `TP == num_key_value_heads == 8` that the packed cache forces
    (`bringup_log/00_MODEL_CARD.md` §4.1) — and `chunk_size` is the **block-cyclic period** in
    tokens, i.e. one `prefill_chunk`.
    """
    _assert_layout_still_shared()
    return _build_shared_table(
        mesh_device=mesh_device,
        kv_cache=kv_cache,
        seq_len=seq_len,
        num_layers=num_layers,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_users=num_users,
        chunk_size=chunk_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )


def build_and_serialize_kv_chunk_table(
    *,
    mesh_device,
    kv_cache,
    seq_len,
    num_layers,
    mesh_shape,
    sp_axis,
    num_users,
    chunk_size,
    num_kv_heads,
    head_dim,
    path,
    first_layer_idx=0,
    num_my_layers=None,
    stage_layout=None,
) -> str:
    """Build the table and serialize it to `path` for the engine to publish. Returns `path`.

    Serialized through `serialize_prebuilt_kv_chunk_table`
    (`models/demos/common/prefill/runners/migration.py:240`) rather than by calling
    `export_to_protobuf_file` directly the way the template does
    (`models/demos/gpt_oss_d_p/tt/runners/kv_chunk_table.py:206`): the shared helper writes
    `<path>.tmp` and `os.replace`s it (`migration.py:39-42`), so a reader polling for the file — and
    the producer does exactly that (`prefill_producer.py:128-134`) — cannot import a half-written table.
    Recipe P10 step 4 names `serialize_kv_chunk_table` instead, but that helper *builds* a
    **single-config** table from a `table_builder` callback (`migration.py:220-237`) and cannot
    express this model's `2 x num_kv_heads` configs; `serialize_prebuilt_kv_chunk_table` is the same
    module's entry point for a table that is already built (`DEC-100`).
    """
    assert_single_rank_stage(
        num_layers=num_layers,
        first_layer_idx=first_layer_idx,
        num_my_layers=num_my_layers,
        stage_layout=stage_layout,
    )
    table = build_kv_chunk_address_table(
        mesh_device=mesh_device,
        kv_cache=kv_cache,
        seq_len=seq_len,
        num_layers=num_layers,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_users=num_users,
        chunk_size=chunk_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    logger.info(
        f"[llama31-8b-d-p-kv-table] {table.num_configs()} configs "
        f"(k_h0..{num_kv_heads - 1}, v_h0..{num_kv_heads - 1}), {table.total_entries()} entries, "
        f"seq_len={seq_len} period={chunk_size} users={num_users} layers={num_layers} -> {path}"
    )
    return serialize_prebuilt_kv_chunk_table(table=table, path=str(path))
