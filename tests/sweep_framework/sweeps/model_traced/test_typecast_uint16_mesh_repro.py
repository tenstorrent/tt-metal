# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Repro test for typecast UINT16 → INT32 PCC failures on mesh devices.

Sweep CI shows UINT16→INT32 typecast fails on mesh (N300 1x2, T3K 1x8) with
PCC 0.1-0.85, but passes on single device.  Aswin's earlier repro using
torch.int16 input passed.  This test isolates:

  1. torch.int32 input (sweep path) vs torch.int16 input (Aswin's path)
  2. ReplicateTensorToMesh vs ShardTensor2dMesh vs no mapper
  3. Single device vs mesh device
  4. Per-device output inspection to identify which device is corrupted

Run on N300:
  pytest tests/sweep_framework/sweeps/model_traced/test_typecast_uint16_mesh_repro.py -v -s

Dispatch via test-dispatch.yaml with:
  command: pytest tests/sweep_framework/sweeps/model_traced/test_typecast_uint16_mesh_repro.py -v -s
  runner-label: ["in-service", "N300"]
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc


# ── helpers ──────────────────────────────────────────────────────────────────


def open_mesh(mesh_shape):
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*mesh_shape),
        l1_small_size=79104,
        dispatch_core_config=ttnn.DispatchCoreConfig(),
    )


def extract_device0(tt_tensor, device):
    """Extract device-0 tensor from mesh, or direct to_torch for single device."""
    if hasattr(device, "get_num_devices"):
        return ttnn.to_torch(ttnn.get_device_tensors(tt_tensor)[0])
    return ttnn.to_torch(tt_tensor)


# ── core test logic ──────────────────────────────────────────────────────────


def run_typecast(device, shape, input_torch_dtype, mesh_mapper=None, pcc_threshold=0.999):
    """
    Create uint16 input → typecast to int32 → compare with golden.

    Args:
        input_torch_dtype: torch.int32 (sweep path) or torch.int16 (Aswin's path)
        mesh_mapper: None (auto), ReplicateTensorToMesh, or ShardTensor2dMesh
    """
    torch.manual_seed(42)

    # Generate input
    raw = torch.randint(0, 65536, shape, dtype=torch.int32).clamp(0, 65535)
    torch_input = raw.to(input_torch_dtype)

    # Golden: the original uint16 values as int32
    if input_torch_dtype == torch.int16:
        golden = torch_input.to(torch.int32) & 0xFFFF  # undo sign-extension
    else:
        golden = raw.to(torch.int32)

    # Send to device as uint16
    from_torch_kwargs = dict(
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if mesh_mapper is not None:
        from_torch_kwargs["mesh_mapper"] = mesh_mapper

    tt_input = ttnn.from_torch(torch_input, **from_torch_kwargs)

    # Typecast uint16 → int32
    tt_output = ttnn.typecast(tt_input, ttnn.int32)

    # Extract output
    output = extract_device0(tt_output, device)

    # Compare
    golden_f32 = golden.to(torch.float32)
    output_f32 = output.to(torch.float32)
    passed, pcc_val = check_with_pcc(golden_f32, output_f32, pcc_threshold)

    mismatch = (golden != output).sum().item()
    total = golden.numel()
    print(f"  shape={shape}, torch_dtype={input_torch_dtype}, mapper={type(mesh_mapper).__name__ if mesh_mapper else 'None'}")
    print(f"  PCC={pcc_val}, mismatches={mismatch}/{total}")

    if mismatch > 0:
        # Show first few mismatches for debugging
        diff_mask = golden != output
        idx = diff_mask.nonzero(as_tuple=False)[:5]
        for i in range(min(5, len(idx))):
            pos = tuple(idx[i].tolist())
            print(f"    [{pos}] golden={golden[pos].item()}, got={output[pos].item()}, "
                  f"raw_input={raw[pos].item()}")

    return passed, pcc_val, mismatch, total


def run_per_device_check(device, shape, input_torch_dtype, mesh_mapper):
    """Check output from EVERY device to find which ones are corrupted."""
    torch.manual_seed(42)

    raw = torch.randint(0, 65536, shape, dtype=torch.int32).clamp(0, 65535)
    torch_input = raw.to(input_torch_dtype)
    if input_torch_dtype == torch.int16:
        golden = torch_input.to(torch.int32) & 0xFFFF
    else:
        golden = raw.to(torch.int32)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
    tt_output = ttnn.typecast(tt_input, ttnn.int32)

    device_tensors = ttnn.get_device_tensors(tt_output)
    results = []
    for i, dt in enumerate(device_tensors):
        out = ttnn.to_torch(dt)
        golden_f32 = golden.to(torch.float32)
        out_f32 = out.to(torch.float32)
        passed, pcc_val = check_with_pcc(golden_f32, out_f32, 0.999)
        mismatch = (golden != out).sum().item()
        print(f"  Device {i}: shape={list(out.shape)}, PCC={pcc_val}, mismatches={mismatch}/{golden.numel()}")
        results.append((i, passed, pcc_val, mismatch))

    return results


# ── Single device tests ──────────────────────────────────────────────────────


class TestSingleDevice:
    """Baseline: single device, no mesh."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = ttnn.open_device(device_id=0, l1_small_size=79104)
        yield
        ttnn.close_device(self.device)

    def test_int32_input_32x64(self):
        """Sweep path: torch.int32 input, shape matching N300 failure."""
        passed, pcc, _, _ = run_typecast(self.device, (1, 1, 32, 64), torch.int32)
        assert passed, f"PCC={pcc}"

    def test_int32_input_32x256(self):
        """Sweep path: torch.int32 input, shape matching T3K failure."""
        passed, pcc, _, _ = run_typecast(self.device, (1, 1, 32, 256), torch.int32)
        assert passed, f"PCC={pcc}"

    def test_int16_input_32x64(self):
        """Aswin's path: torch.int16 input."""
        passed, pcc, _, _ = run_typecast(self.device, (1, 1, 32, 64), torch.int16)
        assert passed, f"PCC={pcc}"

    def test_int16_input_32x256(self):
        """Aswin's path: torch.int16 input."""
        passed, pcc, _, _ = run_typecast(self.device, (1, 1, 32, 256), torch.int16)
        assert passed, f"PCC={pcc}"


# ── Mesh 1x2 (N300) tests ───────────────────────────────────────────────────


class TestMesh1x2:
    """N300-style 1x2 mesh — tests all input dtype × mapper combos."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = open_mesh((1, 2))
        yield
        ttnn.close_mesh_device(self.device)

    # ── ReplicateTensorToMesh (what sweep uses) ──

    def test_replicate_int32_32x64(self):
        mapper = ttnn.ReplicateTensorToMesh(self.device)
        passed, pcc, _, _ = run_typecast(self.device, (1, 1, 32, 64), torch.int32, mapper)
        assert passed, f"PCC={pcc}"

    def test_replicate_int32_32x256(self):
        mapper = ttnn.ReplicateTensorToMesh(self.device)
        passed, pcc, _, _ = run_typecast(self.device, (1, 1, 32, 256), torch.int32, mapper)
        assert passed, f"PCC={pcc}"

    def test_replicate_int16_32x64(self):
        mapper = ttnn.ReplicateTensorToMesh(self.device)
        passed, pcc, _, _ = run_typecast(self.device, (1, 1, 32, 64), torch.int16, mapper)
        assert passed, f"PCC={pcc}"

    # ── ShardTensor2dMesh (traced placement uses Shard(2)) ──

    def test_shard2d_int32_32x64(self):
        mapper = ttnn.ShardTensor2dMesh(self.device, dims=(2, None), mesh_shape=(1, 2))
        passed, pcc, _, _ = run_typecast(self.device, (1, 1, 32, 64), torch.int32, mapper)
        assert passed, f"PCC={pcc}"

    def test_shard2d_int16_32x64(self):
        mapper = ttnn.ShardTensor2dMesh(self.device, dims=(2, None), mesh_shape=(1, 2))
        passed, pcc, _, _ = run_typecast(self.device, (1, 1, 32, 64), torch.int16, mapper)
        assert passed, f"PCC={pcc}"

    # ── No mapper (auto path) ──

    def test_no_mapper_int32_32x64(self):
        passed, pcc, _, _ = run_typecast(self.device, (1, 1, 32, 64), torch.int32)
        assert passed, f"PCC={pcc}"

    # ── Per-device check ──

    def test_per_device_replicate_int32(self):
        """Check each device independently with Replicate + int32."""
        mapper = ttnn.ReplicateTensorToMesh(self.device)
        results = run_per_device_check(self.device, (1, 1, 32, 64), torch.int32, mapper)
        for dev_id, passed, pcc, mismatches in results:
            assert passed, f"Device {dev_id}: PCC={pcc}, mismatches={mismatches}"

    def test_per_device_shard2d_int32(self):
        """Check each device independently with Shard2D + int32."""
        mapper = ttnn.ShardTensor2dMesh(self.device, dims=(2, None), mesh_shape=(1, 2))
        results = run_per_device_check(self.device, (1, 1, 32, 64), torch.int32, mapper)
        for dev_id, passed, pcc, mismatches in results:
            # Note: with sharding, each device gets a slice so golden comparison
            # against full tensor will have mismatches — this is expected.
            # We just log PCC here for diagnostic purposes.
            print(f"  Device {dev_id}: PCC={pcc} (shard — mismatch expected if PCC>0)")


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Isolate specific value ranges to find corruption boundary."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = open_mesh((1, 2))
        yield
        ttnn.close_mesh_device(self.device)

    def test_low_range_only(self):
        """Values [0, 32767] — safe int16 positive range."""
        torch.manual_seed(42)
        shape = (1, 1, 32, 64)
        torch_input = torch.randint(0, 32768, shape, dtype=torch.int32)
        golden = torch_input.to(torch.int32)

        mapper = ttnn.ReplicateTensorToMesh(self.device)
        tt_input = ttnn.from_torch(
            torch_input, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT,
            device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        tt_output = ttnn.typecast(tt_input, ttnn.int32)
        output = extract_device0(tt_output, self.device)

        passed, pcc = check_with_pcc(golden.float(), output.float(), 0.999)
        mismatch = (golden != output).sum().item()
        print(f"  Low range [0,32767]: PCC={pcc}, mismatches={mismatch}/{golden.numel()}")
        assert passed, f"PCC={pcc}"

    def test_high_range_only(self):
        """Values [32768, 65535] — the uint16 range that overflows int16."""
        torch.manual_seed(42)
        shape = (1, 1, 32, 64)
        torch_input = torch.randint(32768, 65536, shape, dtype=torch.int32)
        golden = torch_input.to(torch.int32)

        mapper = ttnn.ReplicateTensorToMesh(self.device)
        tt_input = ttnn.from_torch(
            torch_input, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT,
            device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        tt_output = ttnn.typecast(tt_input, ttnn.int32)
        output = extract_device0(tt_output, self.device)

        passed, pcc = check_with_pcc(golden.float(), output.float(), 0.999)
        mismatch = (golden != output).sum().item()
        print(f"  High range [32768,65535]: PCC={pcc}, mismatches={mismatch}/{golden.numel()}")
        # This may fail due to sign-extension — log for diagnosis
        if not passed:
            print(f"  EXPECTED FAILURE: high range uint16 on mesh — sign-extension bug")
            # Check if values got sign-extended
            diff = output.to(torch.int64) - golden.to(torch.int64)
            if (diff == -65536).all():
                print(f"  Confirmed: all values sign-extended (got value - 65536)")
            elif (diff < 0).any():
                num_neg = (diff < 0).sum().item()
                print(f"  {num_neg}/{golden.numel()} values have negative diff (sign-extension)")
        assert passed, f"PCC={pcc}, mismatches={mismatch}/{golden.numel()}"

    def test_int16_input_with_mask(self):
        """Aswin's original approach: int16 input + & 0xFFFF golden on mesh."""
        torch.manual_seed(42)
        shape = (1, 1, 32, 64)
        raw = torch.randint(0, 65536, shape, dtype=torch.int32)
        torch_input = raw.to(torch.int16)  # values >32767 become negative
        golden = torch_input.to(torch.int32) & 0xFFFF

        mapper = ttnn.ReplicateTensorToMesh(self.device)
        tt_input = ttnn.from_torch(
            torch_input, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT,
            device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        tt_output = ttnn.typecast(tt_input, ttnn.int32)
        output = extract_device0(tt_output, self.device)

        passed, pcc = check_with_pcc(golden.float(), output.float(), 0.999)
        mismatch = (golden != output).sum().item()
        print(f"  int16 input + mask: PCC={pcc}, mismatches={mismatch}/{golden.numel()}")
        assert passed, f"PCC={pcc}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
