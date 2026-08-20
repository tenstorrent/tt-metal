# SPDX-License-Identifier: Apache-2.0
"""Metal 2.0 ProgramSpec probes: compute kernels, DFB memory tricks, precision config.

Everything here is a real on-device run; correctness is the pass/fail. Probes that are *expected*
to fail (validator / compiler errors) assert on the exact message.
"""

import struct

import pytest
import torch

import ttnn

import specs
from specs import ONE_CORE, TILE_BF16, f32_bits


@pytest.fixture(scope="module")
def device():
    ttnn.CONFIG.validate_program_args = True
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def l1_shard(device, shape, dtype, torch_t=None):
    """One shard on core (0,0): the whole tensor is node-local L1 on that core."""
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ONE_CORE, [shape[-2], shape[-1]], ttnn.ShardOrientation.ROW_MAJOR),
    )
    if torch_t is None:
        torch_t = torch.zeros(*shape, dtype=torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)
    return ttnn.from_torch(torch_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype, memory_config=mem)


def dram(device, torch_t, dtype):
    return ttnn.from_torch(torch_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)


def alloc_like(t):
    return ttnn.allocate_tensor_on_device(t.spec, t.device())


# ==================================================================== A: reduce via dfb:: tokens


def test_A_reduce_helper_driven_by_dfb_tokens(device):
    """compute_kernel_lib::reduce<> with DFB tokens in NON-TYPE TEMPLATE PARAMETER position, and
    the dataflow scaler helper likewise. If the implicit constexpr operator uint32_t() did not
    survive a converted constant expression, this would not compile."""
    Ht, Wt = 1, 4
    ta = torch.randn(1, 1, 32 * Ht, 32 * Wt, dtype=torch.bfloat16)
    a = dram(device, ta, ttnn.bfloat16)
    out = dram(device, torch.zeros(1, 1, 32 * Ht, 32), ttnn.bfloat16)

    spec, run, tmap = specs.build_reduce(a, out, Ht, Wt)
    ttnn.generic_op([a, out], spec, run, tmap)

    got = ttnn.to_torch(out).float()[..., 0]
    exp = ta.float().sum(-1)
    assert torch.allclose(got, exp, atol=0.2, rtol=0.02), (got - exp).abs().max()


# ==================================================================== B: eltwise chain


def test_B_eltwise_chain_helper_driven_by_dfb_tokens(device):
    """`square<input(dfb::x), output(dfb::y)>` -- the token must convert inside a constexpr
    function call whose *result* is a class-type template argument."""
    Ht, Wt = 1, 4
    ta = torch.randn(1, 1, 32 * Ht, 32 * Wt, dtype=torch.bfloat16)
    a = dram(device, ta, ttnn.bfloat16)
    out = alloc_like(a)

    spec, run, tmap = specs.build_chain(a, out, Ht, Wt)
    ttnn.generic_op([a, out], spec, run, tmap)

    got = ttnn.to_torch(out).float()
    exp = ta.float() ** 2
    assert torch.allclose(got, exp, atol=0.1, rtol=0.02), (got - exp).abs().max()


# ==================================================================== C: scratchpad on compute


def test_C_scratchpad_on_compute_kernel(device):
    """A compute (TRISC) kernel using a ScratchpadSpec as private, random-access L1."""
    n = 4
    ta = torch.randn(1, 1, 32 * n, 32, dtype=torch.bfloat16)
    a = dram(device, ta, ttnn.bfloat16)
    out = alloc_like(a)

    spec, run, tmap = specs.build_scratchpad(a, out, n, pad_bytes=n * 4)
    ttnn.generic_op([a, out], spec, run, tmap)

    got = ttnn.to_torch(out).float()
    scale = torch.arange(1, n + 1, dtype=torch.float32).repeat_interleave(32).view(1, 1, 32 * n, 1)
    exp = ta.float() * scale
    assert torch.allclose(got, exp, atol=0.1, rtol=0.02), (got - exp).abs().max()


def test_C2_scratchpad_size_zero_rejected(device):
    n = 1
    a = dram(device, torch.randn(1, 1, 32, 32, dtype=torch.bfloat16), ttnn.bfloat16)
    out = alloc_like(a)
    spec, run, tmap = specs.build_scratchpad(a, out, n, pad_bytes=0)
    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        RuntimeError
    ) as e:  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        ttnn.generic_op([a, out], spec, run, tmap)
    print("\n[C2] ", str(e.value)[:400])
    assert "size_per_node == 0" in str(e.value)


def test_C3_scratchpad_cannot_be_shared_by_two_kernels_on_one_core(device):
    """A scratchpad is private per kernel. Two kernels on the same node cannot share one."""
    n = 1
    a = dram(device, torch.randn(1, 1, 32, 32, dtype=torch.bfloat16), ttnn.bfloat16)
    out = alloc_like(a)
    spec, run, tmap = specs.build_scratchpad(a, out, n, pad_bytes=16)
    # Also bind the same scratchpad on the reader (same core).
    spec.kernels[0].scratchpad_bindings = [ttnn.ScratchpadBinding("scale_table", "scale_table")]
    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        RuntimeError
    ) as e:  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        ttnn.generic_op([a, out], spec, run, tmap)
    print("\n[C3] ", str(e.value)[:600])


# ==================================================================== D: aliased DFBs


def test_D1_aliased_in_out_dfb_correctness(device):
    """in_tiles and out_tiles share ONE L1 region. Correct only because tile_regs_commit/wait
    orders the unpack read before the pack write."""
    n = 4
    ta = torch.randn(1, 1, 32 * n, 32, dtype=torch.bfloat16)
    a = dram(device, ta, ttnn.bfloat16)
    out = alloc_like(a)
    spec, run, tmap = specs.build_alias(a, out, n, aliased=True, entries=n)
    ttnn.generic_op([a, out], spec, run, tmap)
    got = ttnn.to_torch(out).float()
    assert torch.equal(got, ta.float()), (got - ta.float()).abs().max()


def test_D2_alias_saves_l1(device):
    """Same spec, N same-size dead DFBs. Non-aliased must OOM at a size the aliased clique fits."""
    n = 1
    a = dram(device, torch.randn(1, 1, 32, 32, dtype=torch.bfloat16), ttnn.bfloat16)
    out = alloc_like(a)

    # N x 96KB of DFB -> over the L1 budget when distinct, one 96KB region when aliased.
    N, ENTRIES = 24, 48  # 48 * 2KB = 96KB each -> 2.25MB distinct vs 96KB aliased

    spec, run, tmap = specs.build_alias_stress(a, out, n, n_scratch_dfbs=N, entries_each=ENTRIES, aliased=False)
    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        RuntimeError
    ) as e:  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        ttnn.generic_op([a, out], spec, run, tmap)
    print("\n[D2 non-aliased OOM] ", str(e.value)[:400])

    spec, run, tmap = specs.build_alias_stress(a, out, n, n_scratch_dfbs=N, entries_each=ENTRIES, aliased=True)
    ttnn.generic_op([a, out], spec, run, tmap)  # must fit
    assert torch.equal(ttnn.to_torch(out).float(), ttnn.to_torch(a).float())


@pytest.mark.parametrize("break_rule", ["not_clique", "size_mismatch", "self_alias", "unknown_name"])
def test_D3_alias_legality_messages(device, break_rule):
    n = 1
    a = dram(device, torch.randn(1, 1, 32, 32, dtype=torch.bfloat16), ttnn.bfloat16)
    out = alloc_like(a)
    spec, run, tmap = specs.build_alias(a, out, n, aliased=True, entries=2)

    if break_rule == "not_clique":
        spec.dataflow_buffers[1].advanced_options = ttnn.DFBAdvancedOptions()  # out drops the back-edge
    elif break_rule == "size_mismatch":
        spec.dataflow_buffers[1].num_entries = 3
    elif break_rule == "self_alias":
        spec.dataflow_buffers[0].advanced_options = ttnn.DFBAdvancedOptions(alias_with=["in_tiles"])
    elif break_rule == "unknown_name":
        spec.dataflow_buffers[0].advanced_options = ttnn.DFBAdvancedOptions(alias_with=["nope"])

    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        RuntimeError
    ) as e:  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        ttnn.generic_op([a, out], spec, run, tmap)
    print(f"\n[D3 {break_rule}] ", str(e.value)[:500])


def test_D4_alias_across_different_kernel_sets_is_allowed(device):
    """Upstream says aliased DFBs must have the same bound kernels. The implemented rule is only
    'same node coverage' -- so two DFBs bound to DIFFERENT kernels on the same core alias fine."""
    n = 1
    a = dram(device, torch.randn(1, 1, 32, 32, dtype=torch.bfloat16), ttnn.bfloat16)
    out = alloc_like(a)
    # pad0 is reader->writer; in_tiles is reader->compute. Different consumer kernels, same node.
    spec, run, tmap = specs.build_alias_stress(a, out, n, n_scratch_dfbs=1, entries_each=2, aliased=False)
    spec.dataflow_buffers[0].advanced_options = ttnn.DFBAdvancedOptions(alias_with=["pad0"])
    spec.dataflow_buffers[2].advanced_options = ttnn.DFBAdvancedOptions(alias_with=["in_tiles"])
    ttnn.generic_op([a, out], spec, run, tmap)
    print("\n[D4] alias across different bound kernels: ACCEPTED")


# ==================================================================== E: fp32 dest + unpack_modes


def test_E1_fp32_dest_without_unpack_mode_is_rejected(device):
    ta = torch.randn(1, 1, 32, 32, dtype=torch.float32)
    a = dram(device, ta, ttnn.float32)
    out = alloc_like(a)
    spec, run, tmap = specs.build_fp32(a, out, 1, enable_32_bit_dest=True, unpack_mode=None)
    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        RuntimeError
    ) as e:  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        ttnn.generic_op([a, out], spec, run, tmap)
    print("\n[E1] ", str(e.value)[:600])
    assert "requires an explicit choice" in str(e.value)


def test_E2_fp32_unpack_to_dest_is_bit_exact(device):
    ta = torch.randn(1, 1, 32, 32, dtype=torch.float32)
    a = dram(device, ta, ttnn.float32)
    out = alloc_like(a)
    spec, run, tmap = specs.build_fp32(a, out, 1, enable_32_bit_dest=True, unpack_mode=ttnn.UnpackMode.UnpackToDest)
    ttnn.generic_op([a, out], spec, run, tmap)
    got = ttnn.to_torch(out)
    print("\n[E2] UnpackToDest max abs err:", (got - ta).abs().max().item())
    assert torch.equal(got, ta)


def test_E3_fp32_unpack_to_src_loses_mantissa(device):
    ta = torch.randn(1, 1, 32, 32, dtype=torch.float32)
    a = dram(device, ta, ttnn.float32)
    out = alloc_like(a)
    spec, run, tmap = specs.build_fp32(a, out, 1, enable_32_bit_dest=True, unpack_mode=ttnn.UnpackMode.UnpackToSrc)
    ttnn.generic_op([a, out], spec, run, tmap)
    got = ttnn.to_torch(out)
    rel = ((got - ta).abs() / ta.abs().clamp_min(1e-6)).max().item()
    print("\n[E3] UnpackToSrc max rel err:", rel, " bit-exact:", torch.equal(got, ta))
    assert torch.allclose(got, ta, rtol=2e-3, atol=1e-6)


def test_E4_fp32_dfb_without_32bit_dest_needs_no_entry(device):
    """The 'explicit choice' demand is conditional on enable_32_bit_dest. Without it, an fp32 DFB
    silently defaults to UnpackToSrc."""
    ta = torch.randn(1, 1, 32, 32, dtype=torch.float32)
    a = dram(device, ta, ttnn.float32)
    out = alloc_like(a)
    spec, run, tmap = specs.build_fp32(a, out, 1, enable_32_bit_dest=False, unpack_mode=None)
    ttnn.generic_op([a, out], spec, run, tmap)
    got = ttnn.to_torch(out)
    print("\n[E4] no-32bit-dest max rel err:", ((got - ta).abs() / ta.abs().clamp_min(1e-6)).max().item())


def test_E5_missing_data_format_on_compute_dfb_is_rejected(device):
    a = dram(device, torch.randn(1, 1, 32, 32, dtype=torch.bfloat16), ttnn.bfloat16)
    out = alloc_like(a)
    spec, run, tmap = specs.build_alias(a, out, 1, aliased=False, entries=2, in_dtype=None)
    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        RuntimeError
    ) as e:  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        ttnn.generic_op([a, out], spec, run, tmap)
    print("\n[E5] ", str(e.value)[:400])
    assert "no data_format_metadata is specified" in str(e.value)


# ==================================================================== F: compute + tensor memory


def test_F1_compute_kernel_cannot_build_a_noc_tensor_accessor(device):
    ta = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    a = dram(device, ta, ttnn.bfloat16)
    out = alloc_like(a)
    scale = l1_shard(device, (1, 1, 32, 32), ttnn.float32, torch.full((1, 1, 32, 32), 3.0, dtype=torch.float32))
    spec, run, tmap = specs.build_compute_tensor_access(a, out, scale, 1, kernel_file="ta_noc_compute.cpp")
    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        Exception
    ) as e:  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        ttnn.generic_op([a, out, scale], spec, run, tmap)
    msg = str(e.value)
    print("\n[F1] ", msg[-1500:])


def test_F2_compute_kernel_reads_resident_l1_via_local_tensor_accessor(device):
    n = 2
    ta = torch.randn(1, 1, 32 * n, 32, dtype=torch.bfloat16)
    a = dram(device, ta, ttnn.bfloat16)
    out = alloc_like(a)
    scale_t = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
    scale_t[..., :, :] = 0.0
    scale_t[0, 0, 0, 0] = 2.5
    scale = l1_shard(device, (1, 1, 32, 32), ttnn.float32, scale_t)

    spec, run, tmap = specs.build_compute_tensor_access(a, out, scale, n, kernel_file="ta_local_compute.cpp")
    ttnn.generic_op([a, out, scale], spec, run, tmap)
    got = ttnn.to_torch(out).float()
    exp = ta.float() * 2.5
    assert torch.allclose(got, exp, atol=0.05, rtol=0.02), (got - exp).abs().max()


def test_F3_borrowed_memory_dfb_needs_a_fake_producer(device):
    """The DFB IS the L1-resident input tensor. No NoC read anywhere; the 'producer' only issues
    credits."""
    n = 4
    ta = torch.randn(1, 1, 32 * n, 32, dtype=torch.bfloat16)
    resident = l1_shard(device, (1, 1, 32 * n, 32), ttnn.bfloat16, ta)
    out = dram(device, torch.zeros(1, 1, 32 * n, 32), ttnn.bfloat16)

    spec, run, tmap = specs.build_borrowed(resident, out, n)
    ttnn.generic_op([resident, out], spec, run, tmap)
    got = ttnn.to_torch(out).float()
    assert torch.equal(got, ta.float()), (got - ta.float()).abs().max()


def test_F4_borrowed_from_a_dram_tensor_is_rejected(device):
    n = 1
    a = dram(device, torch.randn(1, 1, 32, 32, dtype=torch.bfloat16), ttnn.bfloat16)
    out = alloc_like(a)
    spec, run, tmap = specs.build_borrowed(a, out, n)
    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        RuntimeError
    ) as e:  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        ttnn.generic_op([a, out], spec, run, tmap)
    print("\n[F4] ", str(e.value)[:500])
    assert "not L1-resident" in str(e.value)


# ==================================================================== G: self-loop DFB


def test_G_self_loop_dfb_packs_into_resident_output(device):
    n = 4
    ta = torch.randn(1, 1, 32 * n, 32, dtype=torch.bfloat16)
    a = dram(device, ta, ttnn.bfloat16)
    out = l1_shard(device, (1, 1, 32 * n, 32), ttnn.bfloat16)

    spec, run, tmap = specs.build_selfloop(a, out, n)
    ttnn.generic_op([a, out], spec, run, tmap)
    got = ttnn.to_torch(out).float()
    assert torch.equal(got, ta.float()), (got - ta.float()).abs().max()


def test_G2_two_consumer_bindings_under_different_names_rejected(device):
    n = 1
    a = dram(device, torch.randn(1, 1, 32, 32, dtype=torch.bfloat16), ttnn.bfloat16)
    out = alloc_like(a)
    spec, run, tmap = specs.build_alias(a, out, n, aliased=False, entries=2)
    spec.kernels[1].dfb_bindings = list(spec.kernels[1].dfb_bindings) + [ttnn.consumer_of("in_tiles", "in_tiles_again")]
    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        RuntimeError
    ) as e:  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        ttnn.generic_op([a, out], spec, run, tmap)
    print("\n[G2] ", str(e.value)[:600])


# ==================================================================== H: TT_KERNEL on compute


@pytest.mark.parametrize("do_scale", [False, True])
def test_H_tt_kernel_nttp_on_compute(device, do_scale):
    n = 2
    ta = torch.randn(1, 1, 32 * n, 32, dtype=torch.bfloat16)
    a = dram(device, ta, ttnn.bfloat16)
    out = alloc_like(a)
    spec, run, tmap = specs.build_ttkernel(a, out, n, do_scale=do_scale, scale=4.0)
    ttnn.generic_op([a, out], spec, run, tmap)
    got = ttnn.to_torch(out).float()
    exp = ta.float() * (4.0 if do_scale else 1.0)
    assert torch.allclose(got, exp, atol=0.05, rtol=0.02), (got - exp).abs().max()


def test_H2_tt_kernel_cta_name_typo(device):
    n = 1
    a = dram(device, torch.randn(1, 1, 32, 32, dtype=torch.bfloat16), ttnn.bfloat16)
    out = alloc_like(a)
    spec, run, tmap = specs.build_ttkernel(a, out, n, do_scale=True, scale=2.0)
    spec.kernels[1].compile_time_args = {"do_scale": 1, "scale_bitz": f32_bits(2.0)}
    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        Exception
    ) as e:  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        ttnn.generic_op([a, out], spec, run, tmap)
    print("\n[H2] ", str(e.value)[:800])


# ==================================================================== I: matmul_block helper


@pytest.mark.xfail(
    reason="kernel_lib matmul_block_helpers is stale: it calls mm_block_init_short, "
    "which no longer exists in api/compute/matmul.h (renamed to "
    "matmul_block_init, with no _short variant). Not a Metal 2.0 issue.",
    strict=True,
    raises=RuntimeError,
)
def test_I_matmul_block_helper_with_dfb_objects(device):
    """matmul_block takes buffer OBJECTS (deduced `Buf`), not ids, and buffer_compat.hpp already
    has a buf_id(DataflowBuffer) overload -- so this should be a drop-in."""
    ta = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    tb = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    a = dram(device, ta, ttnn.bfloat16)
    b = dram(device, tb, ttnn.bfloat16)
    out = alloc_like(a)
    spec, run, tmap = specs.build_matmul(a, b, out)
    ttnn.generic_op([a, b, out], spec, run, tmap)
    got = ttnn.to_torch(out).float()
    exp = ta.float() @ tb.float()
    assert torch.allclose(got, exp, atol=0.5, rtol=0.05), (got - exp).abs().max()


# ==================================================================== binding-surface gotchas


def test_J_spec_vector_item_assignment_is_silently_lost(device):
    """`spec.dataflow_buffers` casts to a NEW Python list of REFERENCES: mutating an element's
    field reaches the C++ spec, but rebinding a slot does not. No error either way."""
    a = dram(device, torch.randn(1, 1, 32, 32, dtype=torch.bfloat16), ttnn.bfloat16)
    out = alloc_like(a)
    spec, _, _ = specs.build_alias(a, out, 1, aliased=False, entries=2)

    h0 = ttnn.compute_program_spec_hash(spec)

    spec.dataflow_buffers[0] = ttnn.DataflowBufferSpec(
        unique_id="in_tiles", entry_size=TILE_BF16, num_entries=99, data_format=ttnn.bfloat16
    )
    h_after_slot_assign = ttnn.compute_program_spec_hash(spec)

    spec.dataflow_buffers[0].num_entries = 99
    h_after_field_assign = ttnn.compute_program_spec_hash(spec)

    print("\n[J] slot assignment changed the spec:", h_after_slot_assign != h0)
    print("[J] field assignment changed the spec:", h_after_field_assign != h0)
    assert h_after_slot_assign == h0, "slot assignment unexpectedly took effect"
    assert h_after_field_assign != h0, "field assignment unexpectedly lost"


def test_I2_raw_matmul_api_with_dfb_tokens(device):
    """The raw compute matmul API driven from dfb:: tokens."""
    ta = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    tb = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    a = dram(device, ta, ttnn.bfloat16)
    b = dram(device, tb, ttnn.bfloat16)
    out = alloc_like(a)
    spec, run, tmap = specs.build_matmul(a, b, out, kernel_file="matmul_raw_compute.cpp")
    ttnn.generic_op([a, b, out], spec, run, tmap)
    got = ttnn.to_torch(out).float()
    exp = ta.float() @ tb.float()
    assert torch.allclose(got, exp, atol=0.5, rtol=0.05), (got - exp).abs().max()


def test_C4_compute_scratchpad_is_shared_by_all_three_triscs(device):
    """UNPACK writes, MATH reads -- same private L1 region, no per-thread scratchpad."""
    n = 2
    ta = torch.randn(1, 1, 32 * n, 32, dtype=torch.bfloat16)
    a = dram(device, ta, ttnn.bfloat16)
    out = alloc_like(a)
    spec, run, tmap = specs.build_scratchpad(a, out, n, pad_bytes=16, kernel_file="scratch_threads_compute.cpp")
    ttnn.generic_op([a, out], spec, run, tmap)
    got = ttnn.to_torch(out).float()
    exp = ta.float() * 3.0
    print("\n[C4] UNPACK's sentinel was visible to MATH:", torch.allclose(got, exp, atol=0.1, rtol=0.02))
    assert torch.allclose(got, exp, atol=0.1, rtol=0.02), (got - exp).abs().max()


def test_C5_oversized_scratchpad_hits_the_same_l1_budget(device):
    a = dram(device, torch.randn(1, 1, 32, 32, dtype=torch.bfloat16), ttnn.bfloat16)
    out = alloc_like(a)
    spec, run, tmap = specs.build_scratchpad(a, out, 1, pad_bytes=2 * 1024 * 1024)
    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        RuntimeError
    ) as e:  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        ttnn.generic_op([a, out], spec, run, tmap)
    print("\n[C5] ", str(e.value)[:400])


def test_K_compute_kernel_cannot_bind_a_semaphore(device):
    a = dram(device, torch.randn(1, 1, 32, 32, dtype=torch.bfloat16), ttnn.bfloat16)
    out = alloc_like(a)
    spec, run, tmap = specs.build_alias(a, out, 1, aliased=False, entries=2)
    spec.semaphores = [ttnn.SemaphoreSpec(unique_id="sem", target_nodes=ONE_CORE)]
    spec.kernels[1].semaphore_bindings = [ttnn.SemaphoreBinding("sem", "sem")]
    with pytest.raises(  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        RuntimeError
    ) as e:  # allow-pytest.raises: probe asserts on a host-validator rejection and captures its text
        ttnn.generic_op([a, out], spec, run, tmap)
    print("\n[K] ", str(e.value)[:400])


def test_L_validation_off_lets_an_illegal_spec_through(device):
    """ttnn.CONFIG.validate_program_args is the ONLY thing standing between you and a silently
    misconfigured program. With it off, the fp32/unpack_modes demand from E1 disappears."""
    ta = torch.randn(1, 1, 32, 32, dtype=torch.float32)
    a = dram(device, ta, ttnn.float32)
    out = alloc_like(a)
    spec, run, tmap = specs.build_fp32(a, out, 1, enable_32_bit_dest=True, unpack_mode=None)
    ttnn.CONFIG.validate_program_args = False
    print("\n[L] flag readback right before the call:", ttnn.CONFIG.validate_program_args)
    try:
        ttnn.generic_op([a, out], spec, run, tmap)
        got = ttnn.to_torch(out)
        print(
            "\n[L] validation OFF: illegal spec RAN. max rel err:",
            ((got - ta).abs() / ta.abs().clamp_min(1e-6)).max().item(),
        )
        ran = True
    except RuntimeError as e:
        print("\n[L] validation OFF still rejected: ", str(e)[:300])
        ran = False
    finally:
        ttnn.CONFIG.validate_program_args = True
    # Recorded, not asserted: see FINDINGS.
    print("[L] ran without validation:", ran)


# ==================================================================== M: DFB format plumbing


def test_M1_bf16_in_fp32_out_needs_no_kernel_change(device):
    """copy_compute.cpp is byte-identical to the bf16->bf16 case: the packer's output format comes
    from the named DFB, not from anything the kernel says."""
    n = 2
    ta = torch.randn(1, 1, 32 * n, 32, dtype=torch.bfloat16)
    a = dram(device, ta, ttnn.bfloat16)
    out = dram(device, torch.zeros(1, 1, 32 * n, 32, dtype=torch.float32), ttnn.float32)
    spec, run, tmap = specs.build_mixed_format(
        a,
        out,
        n,
        in_entry=specs.TILE_BF16,
        out_entry=specs.TILE_F32,
        in_fmt=ttnn.bfloat16,
        out_fmt=ttnn.float32,
    )
    ttnn.generic_op([a, out], spec, run, tmap)
    got = ttnn.to_torch(out)
    assert torch.equal(got, ta.float()), (got - ta.float()).abs().max()


def test_M2_lying_about_the_dfb_format_is_not_caught(device):
    """The DFB's data_format is a user assertion about the bytes. Declaring fp32 for a bf16 tensor
    is accepted by the validator and silently produces garbage."""
    n = 1
    ta = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    a = dram(device, ta, ttnn.bfloat16)
    out = alloc_like(a)
    spec, run, tmap = specs.build_mixed_format(
        a,
        out,
        n,
        in_entry=specs.TILE_BF16,
        out_entry=specs.TILE_BF16,
        in_fmt=ttnn.float32,
        out_fmt=ttnn.bfloat16,  # <-- in_tiles is really bf16
    )
    ttnn.generic_op([a, out], spec, run, tmap)
    got = ttnn.to_torch(out).float()
    ok = torch.allclose(got, ta.float(), atol=0.05, rtol=0.05)
    print("\n[M2] declared fp32 over bf16 bytes -> output still correct:", ok)
    print("[M2] max abs diff:", (got - ta.float()).abs().max().item())
