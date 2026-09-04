# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The KV-chunk **address table**, isolated from every numerical question. Gate: `G-KV-TABLE`.

**What this gate exists for.** `G-MOCK-MIG` produces one PCC over one slot, and one PCC cannot
separate a wrong address table from a numerical problem: a table that reads the *next* head's column
still scores 0.99890 on a position-labelled probe (`G-KV-TP8`, recipe §2.5). So a mapping claim is
gated on **bit-equality**, never correlation — `torch.equal`, `rtol = atol = 0`
(`BRINGUP_RECIPE.md:1962-1967`).

Four claims, each with its own control:

1. **position -> address.** Every chunk the table addresses, read back over the **same UMD path the
   migration worker and the producer use** (`ttnn.experimental.disaggregation.read_dram_umd`), is
   bit-identical to that chunk read through `ttnn.to_torch` on the live device tensor.
2. **head -> config -> chip.** Config `h`'s device group for SP row `r` is exactly the chip at
   `MeshCoordinate(r, h)` — compared on fabric-node ids, not on values.
3. **K/V separation.** Configs `0..N-1` address K and `N..2N-1` address V, and the two are
   distinguishable because the probe labels them.
4. **the protobuf round trip.** Export, re-import, and every lookup — address, size, device group
   **and config name** — survives. The name matters: `import_from_protobuf` rebuilds configs through
   a `std::map`, so unpadded names would order `"10"` before `"2"` and silently renumber 16 configs
   (`models/demos/gpt_oss_d_p/tt/runners/kv_chunk_table.py:20-23`).

**The probe.** Each 32-lane field of a head is a constant label, so every one of `bfloat8_b`'s
16-element exponent blocks is homogeneous and the value survives the dtype exactly. Every label is
**< 128**, which is `bfloat8_b`'s exact-integer ceiling and *not* the recipe's blanket 256 — that is
the **bf16** ceiling (`DEC-044`, `07_RISKS.md` R-013). A probe written to the stated rule fails on a
correct cache.

**Mesh:** the deployment `(4, 8)`. Appendix A gives this gate "target mesh", and it must be: the
table's whole content is the SP x TP geometry, and `G-KV`'s `(1,1)` arm exercises a head count the
model never emits (`R-001`). A top-level partial mesh dies in fabric bring-up on this galaxy
(`BRINGUP_RECIPE.md:1672-1693`), so this opens the full mesh and shards nothing smaller.

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_kv_chunk_table.py -x -q
"""

import os

# The producer resolves its adapter at module import (`prefill_producer.py:89`), so the model must be
# selected BEFORE that import below — otherwise it pulls the registry default (`kimi_k2_7`) and its
# whole MLA stack. Set, not overridden: a caller running this under another PREFILL_MODEL should see
# the mismatch rather than have it silently corrected.
os.environ.setdefault("PREFILL_MODEL", "llama31_8b_d_p")

import pytest  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

import ttnn  # noqa: E402
from models.demos.common.prefill.runners.prefill_producer import _decode_bfp8_chunk, _resolve_unique_id  # noqa: E402
from models.demos.llama31_8b_d_p.tests.test_factory import (  # noqa: E402
    GALAXY_MESH_SHAPE,
    galaxy_device_params,
    requires_galaxy,
)
from models.demos.llama31_8b_d_p.tt.attention.kv_cache import (  # noqa: E402
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    allocate_kv_cache,
    write_kv_chunk,
)
from models.demos.llama31_8b_d_p.tt.runners.kv_chunk_table import (  # noqa: E402
    build_and_serialize_kv_chunk_table,
    build_kv_chunk_address_table,
)

HEAD_DIM = 128
SP_AXIS = 0
TP_AXIS = 1
SEQ_LEN = 512
NUM_USERS = 2
NUM_LAYERS = 2

# `bfloat8_b`'s exact-integer ceiling, measured (`DEC-044`). Every probe label is strictly below it,
# so a failing probe cannot be the probe's own numerics.
BF8_EXACT_INTEGER_CEILING = 128

# The four 32-lane label fields of a head. 32 is a multiple of bf8_b's 16-element exponent block, so
# every block is homogeneous and its shared exponent is exact.
_FIELD = HEAD_DIM // 4


def _probe_head(position_start, seq_len, *, head, slot, layer, is_k) -> torch.Tensor:
    """`[seq_len, HEAD_DIM]` whose every 32-lane field names one coordinate of this chunk.

    Fully determines `(position, head, slot, layer, K-or-V)`, so an address error cannot land on
    another chunk and still decode to the same bytes:

    * lanes   0..31  -> `position % 128`
    * lanes  32..63  -> `position // 128 + 1`
    * lanes  64..95  -> `head + 1`, plus 16 for V (so K head 0 and V head 0 differ)
    * lanes  96..127 -> `slot * num_layers + layer + 1` (the cache's own user-major batch index + 1)
    """
    out = torch.empty(seq_len, HEAD_DIM)
    batch_label = slot * NUM_LAYERS + layer + 1
    head_label = head + 1 + (0 if is_k else 16)
    assert max(batch_label, head_label) < BF8_EXACT_INTEGER_CEILING
    for row in range(seq_len):
        position = position_start + row
        low = position % BF8_EXACT_INTEGER_CEILING
        high = position // BF8_EXACT_INTEGER_CEILING + 1
        assert high < BF8_EXACT_INTEGER_CEILING, f"position {position} overflows the probe's high field"
        out[row, 0:_FIELD] = float(low)
        out[row, _FIELD : 2 * _FIELD] = float(high)
        out[row, 2 * _FIELD : 3 * _FIELD] = float(head_label)
        out[row, 3 * _FIELD :] = float(batch_label)
    return out


def _write_probe(mesh_device, kv_cache, *, num_kv_heads, period):
    """Fill every `(slot, layer)` with the labelled probe, through the real `write_kv_chunk`.

    Written in `SEQ_LEN // period` chunks with `kv_actual` advancing, so the cache's block-cyclic
    layout is the one a chunked prefill of that period produces — the layout the table must describe.
    Returns `{(slot, layer, is_k): [SEQ_LEN, num_kv_heads, HEAD_DIM]}`, the host-side truth.
    """
    rows, cols = tuple(mesh_device.shape)
    in_dims = [None, None]
    in_dims[SP_AXIS] = 2  # sequence over the SP rows
    in_dims[TP_AXIS] = 1  # one KV head per TP column

    truth = {}
    for slot in range(kv_cache.num_users):
        for layer in range(kv_cache.num_layers):
            for is_k in (True, False):
                truth[(slot, layer, is_k)] = torch.stack(
                    [
                        _probe_head(0, SEQ_LEN, head=head, slot=slot, layer=layer, is_k=is_k)
                        for head in range(num_kv_heads)
                    ],
                    dim=1,
                )

    for slot in range(kv_cache.num_users):
        for layer in range(kv_cache.num_layers):
            for start in range(0, SEQ_LEN, period):
                k_host = truth[(slot, layer, True)][start : start + period].permute(1, 0, 2).unsqueeze(0)
                v_host = truth[(slot, layer, False)][start : start + period].permute(1, 0, 2).unsqueeze(0)
                tensors = [
                    ttnn.from_torch(
                        host,
                        device=mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=in_dims),
                    )
                    for host in (k_host, v_host)
                ]
                write_kv_chunk(
                    kv_cache,
                    tensors[0],
                    tensors[1],
                    slot_idx=slot,
                    layer_idx=layer,
                    kv_actual=start,
                    sp_axis=SP_AXIS,
                )
                for tensor in tensors:
                    tensor.deallocate(True)
    ttnn.synchronize_device(mesh_device)
    return truth


def _device_map(mesh_device) -> dict:
    """`{(mesh_id, chip_id): unique_id}` — the sidecar the runner publishes, built in-process.

    Same construction as `models/demos/common/prefill/runners/migration.py:142-151`
    (`_enumerate_devices`), whose output `serialize_device_map` writes and the producer reads back
    (`prefill_producer.py:151`).
    """
    rows, cols = tuple(mesh_device.shape)
    out = {}
    for row in range(rows):
        for col in range(cols):
            fnid = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(row, col))
            unique_id = int(ttnn.cluster.get_chip_unique_id_from_fabric_node_id(int(fnid.mesh_id), int(fnid.chip_id)))
            out[(int(fnid.mesh_id), int(fnid.chip_id))] = unique_id
    return out


def _read_through_table(table, device_map, *, layer, position, slot, config_id) -> torch.Tensor:
    """One 32-token chunk, read over UMD exactly the way the producer's read-back does.

    `prefill_producer.py:535-538`: look the location up, resolve the device group's fabric node to a
    UMD unique id, `read_dram_umd`, decode the `bfloat8_b` tile bytes.
    """
    location = table.lookup(layer, position, slot, config_id)
    unique_id = _resolve_unique_id(table.get_device_group(location.device_group_index).fabric_node_ids, device_map)
    raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, location.noc_addr, location.size_bytes)
    assert len(raw) == location.size_bytes
    return _decode_bfp8_chunk(raw, HEAD_DIM)


def _local_row_of(position, *, period, sp) -> tuple:
    """`(sp_row, local_row)` — the inverse of the block-cyclic map, for the live-cache comparison.

    `tests/galaxy_prefill_kv_pcc.py:152` (`cache_row_to_global_position`) is the forward direction;
    this is its inverse, and the two are asserted to agree below rather than trusted.
    """
    tokens_per_period_local = period // sp
    seq_chunk, offset = divmod(position, period)
    sp_row, local_in_period = divmod(offset, tokens_per_period_local)
    return sp_row, seq_chunk * tokens_per_period_local + local_in_period


def _live_cache_chunk(cache, mesh_device, *, slot, layer, head, position, period) -> torch.Tensor:
    """The same 32 tokens read through `ttnn.to_torch` on the chip that owns them."""
    rows, cols = tuple(mesh_device.shape)
    sp_row, local_row = _local_row_of(position, period=period, sp=rows)
    batch = slot * NUM_LAYERS + layer
    host = ttnn.to_torch(ttnn.get_device_tensors(cache)[sp_row * cols + head]).float()
    return host[batch, 0, local_row : local_row + NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, :]


def _build(mesh_device, kv_cache, *, num_kv_heads, period):
    return build_kv_chunk_address_table(
        mesh_device=mesh_device,
        kv_cache=kv_cache,
        seq_len=SEQ_LEN,
        num_layers=NUM_LAYERS,
        mesh_shape=GALAXY_MESH_SHAPE,
        sp_axis=SP_AXIS,
        num_users=NUM_USERS,
        chunk_size=period,
        num_kv_heads=num_kv_heads,
        head_dim=HEAD_DIM,
    )


@pytest.fixture
def probed_cache(mesh_device, period):
    """The `(4,8)` cache filled with the labelled probe at this `period`, plus the host truth.

    **Function-scoped, and it has to be.** The repo's `mesh_device` fixture is function-scoped, so
    the mesh is closed and reopened between tests; a cached device tensor then belongs to a closed
    mesh and the next `ttnn.to_torch` on it aborts with
    `TT_FATAL @ tt_metal/distributed/mesh_device.cpp:845: id < mesh_command_queues_.size()` /
    "cq_id 0 is out of range" — a message naming neither the fixture nor the stale tensor. The
    first draft of this file cached the probe across the module and failed exactly there
    (`DEC-105`). Note that a stale tensor's `buffer_address()` still returns a plausible number, so
    the address-only tests passed while the read-back ones did not.
    """
    num_kv_heads = GALAXY_MESH_SHAPE[TP_AXIS]
    cache = allocate_kv_cache(
        mesh_device,
        num_layers=NUM_LAYERS,
        max_seq_len=SEQ_LEN,
        sp_axis=SP_AXIS,
        num_users=NUM_USERS,
        head_dim=HEAD_DIM,
    )
    truth = _write_probe(mesh_device, cache, num_kv_heads=num_kv_heads, period=period)
    return cache, truth, num_kv_heads


# `period == SEQ_LEN` is the one-shot layout (the whole prompt in one chunk); `SEQ_LEN // 4` is the
# chunked one, four `prefill_chunk` calls with `kv_actual` advancing. Both must be describable, and
# reading the cache at two different block-cyclic periods is itself the structural control
# `G-MESH-KV` uses: a walk with the wrong period cannot pass both.
PERIODS = [pytest.param(SEQ_LEN, id="period512"), pytest.param(SEQ_LEN // 4, id="period128")]


@requires_galaxy
@pytest.mark.parametrize("device_params", [galaxy_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("period", PERIODS)
@pytest.mark.timeout(0)
@torch.no_grad()
def test_the_block_cyclic_map_and_its_inverse_agree(mesh_device, period):
    """Precondition: this file's inverse map covers every global position exactly once.

    Without it a wrong inverse could make the live-cache comparison below compare the wrong rows on
    *both* sides and pass. Checked against `tests/galaxy_prefill_kv_pcc.py`'s forward map, which
    `G-MESH-KV` and `G-RACE` already stand on.
    """
    from models.demos.llama31_8b_d_p.tests.galaxy_prefill_kv_pcc import cache_row_to_global_position

    sp = GALAXY_MESH_SHAPE[SP_AXIS]
    seen = set()
    for position in range(SEQ_LEN):
        sp_row, local_row = _local_row_of(position, period=period, sp=sp)
        forward = cache_row_to_global_position(local_row, sp_row, chunk_local=period // sp, chunk_global=period)
        assert forward == position, f"inverse map disagrees at position {position}: got row {sp_row}/{local_row}"
        seen.add((sp_row, local_row))
    assert len(seen) == SEQ_LEN, "the inverse map is not injective"


@requires_galaxy
@pytest.mark.parametrize("device_params", [galaxy_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("period", PERIODS)
@pytest.mark.timeout(0)
@torch.no_grad()
def test_table_geometry(mesh_device, probed_cache, period):
    """Config count, entry count and chunk size, from the arithmetic rather than from the builder."""
    cache, _truth, num_kv_heads = probed_cache
    table = _build(mesh_device, cache, num_kv_heads=num_kv_heads, period=period)

    expected_chunk_bytes = (HEAD_DIM // 32) * 1088  # bf8_b tile: 1024 mantissa + 64 exponent bytes
    positions = SEQ_LEN // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    assert table.num_configs() == 2 * num_kv_heads == 16
    assert table.total_entries() == 2 * num_kv_heads * NUM_LAYERS * NUM_USERS * positions
    for config_id in range(table.num_configs()):
        config = table.config(config_id)
        assert config.chunk_size_bytes == expected_chunk_bytes
        assert config.chunk_n_tokens == NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
        assert config.num_layers == NUM_LAYERS and config.num_slots == NUM_USERS
        assert config.max_sequence_length == SEQ_LEN
    logger.info(
        f"[G-KV-TABLE] period={period}: {table.num_configs()} configs, {table.total_entries()} entries, "
        f"{expected_chunk_bytes} B/chunk, {positions} positions x {NUM_LAYERS} layers x {NUM_USERS} slots"
    )


@requires_galaxy
@pytest.mark.parametrize("device_params", [galaxy_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("period", PERIODS)
@pytest.mark.timeout(0)
@torch.no_grad()
def test_head_to_config_to_chip(mesh_device, probed_cache, period):
    """**Claim 2.** Config `h`'s group for SP row `r` is the chip at `MeshCoordinate(r, h)`.

    Compared on fabric-node ids, not on tensor values: this is a mapping claim, and a mapping claim
    gated on PCC passes on a rotated map (`G-KV-TP8`, recipe §2.5). Each group is a **single**
    chip — one KV head per column — so a group with two members would mean the table thinks the head
    is replicated.
    """
    cache, _truth, num_kv_heads = probed_cache
    table = _build(mesh_device, cache, num_kv_heads=num_kv_heads, period=period)
    sp = GALAXY_MESH_SHAPE[SP_AXIS]

    for config_id in range(table.num_configs()):
        head = config_id % num_kv_heads
        for position in range(0, SEQ_LEN, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
            expected_row, _ = _local_row_of(position, period=period, sp=sp)
            location = table.lookup(0, position, 0, config_id)
            group = table.get_device_group(location.device_group_index).fabric_node_ids
            assert len(group) == 1, f"config {config_id} position {position} has {len(group)} chips, not one"
            expected = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(expected_row, head))
            assert (int(group[0].mesh_id), int(group[0].chip_id)) == (int(expected.mesh_id), int(expected.chip_id)), (
                f"config {config_id} (head {head}) position {position} resolves to chip "
                f"({int(group[0].mesh_id)},{int(group[0].chip_id)}) but head {head} on SP row "
                f"{expected_row} is ({int(expected.mesh_id)},{int(expected.chip_id)})"
            )


@requires_galaxy
@pytest.mark.parametrize("device_params", [galaxy_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("period", PERIODS)
@pytest.mark.timeout(0)
@torch.no_grad()
def test_every_addressed_chunk_is_bit_identical_to_the_live_cache(mesh_device, probed_cache, period):
    """**Claims 1 and 3.** Every entry, read over UMD, `torch.equal` to the live device tensor.

    Two comparands, and both matter:

    * the **live cache** through `ttnn.to_torch` — pure addressing, with no dtype question at all,
      because both sides are the same bytes;
    * the **host probe**, which says the bytes at that address really are that
      `(head, position, slot, layer, K/V)` and not merely self-consistent.
    """
    cache, truth, num_kv_heads = probed_cache
    table = _build(mesh_device, cache, num_kv_heads=num_kv_heads, period=period)
    device_map = _device_map(mesh_device)

    compared = 0
    for config_id in range(table.num_configs()):
        is_k = config_id < num_kv_heads
        head = config_id if is_k else config_id - num_kv_heads
        tensor = cache.k if is_k else cache.v
        for slot in range(NUM_USERS):
            for layer in range(NUM_LAYERS):
                for position in range(0, SEQ_LEN, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                    read = _read_through_table(
                        table, device_map, layer=layer, position=position, slot=slot, config_id=config_id
                    )
                    live = _live_cache_chunk(
                        tensor, mesh_device, slot=slot, layer=layer, head=head, position=position, period=period
                    )
                    assert torch.equal(read.float(), live), (
                        f"config {config_id} (head {head}, {'K' if is_k else 'V'}) slot {slot} layer {layer} "
                        f"position {position}: the address the table gives is not the chunk the live "
                        f"cache holds (max|delta| = {(read.float() - live).abs().max().item()})"
                    )
                    expected = truth[(slot, layer, is_k)][
                        position : position + NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head, :
                    ]
                    assert torch.equal(read.float(), expected), (
                        f"config {config_id} slot {slot} layer {layer} position {position} decodes to "
                        f"the wrong labels: got {read[0, ::_FIELD].tolist()}, "
                        f"expected {expected[0, ::_FIELD].tolist()}"
                    )
                    compared += 1
    assert compared == table.total_entries()
    logger.info(f"[G-KV-TABLE] period={period}: {compared} chunks bit-identical over UMD (torch.equal, rtol=atol=0)")


@requires_galaxy
@pytest.mark.parametrize("device_params", [galaxy_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("period", [pytest.param(SEQ_LEN // 4, id="period128")])
@pytest.mark.timeout(0)
@torch.no_grad()
def test_read_device_chunk_agrees_with_the_umd_path(mesh_device, probed_cache, period):
    """A second, independent resolution of the same entry: the table's own `read_device_chunk`.

    It resolves the device through the global ControlPlane rather than through the device-map
    sidecar, so agreement means the sidecar the runner publishes and the control plane's own view
    name the same chip — the thing that silently vanishes when the sidecar is stale
    (`PREFILL_MIGRATION_TESTING.md:449-454`).
    """
    cache, _truth, num_kv_heads = probed_cache
    table = _build(mesh_device, cache, num_kv_heads=num_kv_heads, period=period)
    device_map = _device_map(mesh_device)

    for config_id in (0, num_kv_heads - 1, num_kv_heads, 2 * num_kv_heads - 1):
        for position in (0, SEQ_LEN - NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
            over_umd = _read_through_table(table, device_map, layer=1, position=position, slot=1, config_id=config_id)
            over_control_plane = _decode_bfp8_chunk(
                table.read_device_chunk(layer=1, position=position, slot=1, config_id=config_id), HEAD_DIM
            )
            assert torch.equal(over_umd, over_control_plane), (
                f"config {config_id} position {position}: the device-map path and the control-plane "
                f"path disagree, so one of them is resolving the wrong chip"
            )


@requires_galaxy
@pytest.mark.parametrize("device_params", [galaxy_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("period", [pytest.param(SEQ_LEN // 4, id="period128")])
@pytest.mark.timeout(0)
@torch.no_grad()
def test_negative_controls_every_confusable_lookup_must_differ(mesh_device, probed_cache, period):
    """**The controls.** Four wrong lookups, each of which a correct table must NOT satisfy.

    The recipe names the first ("a negative control that reads one head through another's config",
    `BRINGUP_RECIPE.md:1965`); the other three cover the coordinates the same walk could confuse.
    Every one is checked with `torch.equal` — the discriminator a mapping bug needs, since the
    rotated-head case still scores PCC 0.99890 (recipe §2.5).
    """
    cache, _truth, num_kv_heads = probed_cache
    table = _build(mesh_device, cache, num_kv_heads=num_kv_heads, period=period)
    device_map = _device_map(mesh_device)

    def read(**kwargs):
        call = {"layer": 0, "position": 0, "slot": 0, "config_id": 0}
        call.update(kwargs)
        return _read_through_table(table, device_map, **call)

    correct = read()
    controls = {
        # 1. one head through the NEXT head's config (the recipe's named control).
        "rotated_head": read(config_id=1),
        # 2. K's chunk through V's config for the same head.
        "k_through_v": read(config_id=num_kv_heads),
        # 3. layer 0 through layer 1's row — the user-major packing's neighbour.
        "next_layer": read(layer=1),
        # 4. the next 32-token block, i.e. one position chunk along.
        "next_position": read(position=NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK),
        # 5. slot 0 through slot 1 — the users share one cache tensor.
        "next_slot": read(slot=1),
    }
    for name, wrong in controls.items():
        assert not torch.equal(correct, wrong), (
            f"control {name!r} read the SAME bytes as the correct lookup, so this gate cannot tell "
            f"the two apart and its positive result proves nothing (recipe section 2.5)"
        )
        logger.info(
            f"[G-KV-TABLE] control {name}: differs from the correct chunk "
            f"(max|delta| = {(correct - wrong).abs().max().item():.1f})"
        )


@requires_galaxy
@pytest.mark.parametrize("device_params", [galaxy_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("period", [pytest.param(SEQ_LEN // 4, id="period128")])
@pytest.mark.timeout(0)
@torch.no_grad()
def test_protobuf_round_trip_preserves_every_lookup_and_every_config_name(mesh_device, probed_cache, period, tmp_path):
    """**Claim 4.** The file the engine publishes is what the reader gets back.

    Config **names** are checked as well as addresses, because `import_from_protobuf` rebuilds
    configs through a `std::map`: with 16 configs, unpadded names would order `"10"` before `"2"`
    and renumber every config id, so the producer would look K head 2 up and get K head 10
    (`models/demos/gpt_oss_d_p/tt/runners/kv_chunk_table.py:20-23`). 16 > 10, so this model is on
    the wrong side of that boundary and the guard is load-bearing here rather than theoretical.
    """
    cache, _truth, num_kv_heads = probed_cache
    original = _build(mesh_device, cache, num_kv_heads=num_kv_heads, period=period)

    path = tmp_path / "llama31_8b_d_p_kv_chunk_table.pb"
    returned = build_and_serialize_kv_chunk_table(
        mesh_device=mesh_device,
        kv_cache=cache,
        seq_len=SEQ_LEN,
        num_layers=NUM_LAYERS,
        mesh_shape=GALAXY_MESH_SHAPE,
        sp_axis=SP_AXIS,
        num_users=NUM_USERS,
        chunk_size=period,
        num_kv_heads=num_kv_heads,
        head_dim=HEAD_DIM,
        path=str(path),
    )
    assert returned == str(path), "the engine uses the returned path as the one it publishes"
    assert path.is_file() and path.stat().st_size > 0
    assert not (tmp_path / f"{path.name}.tmp").exists(), (
        "the .tmp staging file survived; serialize_prebuilt_kv_chunk_table os.replace()s it "
        "(migration.py:39-42) so a polling reader cannot see a half-written table"
    )

    restored = ttnn.experimental.disaggregation.import_from_protobuf_file(str(path))
    assert restored.num_configs() == original.num_configs()
    assert restored.total_entries() == original.total_entries()
    for config_id in range(original.num_configs()):
        assert restored.config_name(config_id) == original.config_name(config_id), (
            f"config {config_id} came back named {restored.config_name(config_id)!r}, was "
            f"{original.config_name(config_id)!r} — the std::map ordering trap"
        )
        assert (
            restored.config_name(config_id) == f"{config_id:02d}"
        ), "config names must be zero-padded so lexicographic map order == numeric config_id order"
        assert restored.config(config_id).chunk_size_bytes == original.config(config_id).chunk_size_bytes
        for slot in range(NUM_USERS):
            for layer in range(NUM_LAYERS):
                for position in range(0, SEQ_LEN, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                    before = original.lookup(layer, position, slot, config_id)
                    after = restored.lookup(layer, position, slot, config_id)
                    assert (after.noc_addr, after.size_bytes) == (before.noc_addr, before.size_bytes)
                    assert int(after.device_group_index) == int(before.device_group_index)
    logger.info(
        f"[G-KV-TABLE] protobuf round trip: {restored.num_configs()} configs, "
        f"{restored.total_entries()} entries, names {[restored.config_name(i) for i in range(4)]}... preserved"
    )
