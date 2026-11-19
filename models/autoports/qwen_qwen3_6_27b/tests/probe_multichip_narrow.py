# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Test-only local-head narrow correction candidate; production decoder untouched.

The normal multichip runner still owns PCC, cache, trace, stress and performance
oracles. --narrow and --l1 independently override narrow_gdn_correction and
correction_l1; the same policy keys work through --policy-json/--policy-file.
"""

import argparse
import hashlib
import json
import re
import shlex
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests import run_multichip_decoder as runner
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import (
    _NARROW_GDN_COMBINE_COMPUTE,
    _NARROW_GDN_COMBINE_READER,
    _NARROW_GDN_COMBINE_WRITER,
)


def _specialize(source, expected):
    """Replace whole integer tokens and fail closed if upstream kernel changes."""
    substitutions = {516: 130, 320: 80, 512: 128, 192: 48}
    for old, count in expected.items():
        source, found = re.subn(r"\b" + str(old) + r"\b", str(substitutions[old]), source)
        assert found == count, (old, found, count)
    for old in substitutions:
        assert re.search(r"\b" + str(old) + r"\b", source) is None, old
    return source


# Output columns: QKV tiles[0,80), Z[80,128), BA[128,130).
# Narrow residual columns: QKV[0,80), BA[80,82). Runtime pn/rn retain
# the true padded strides (normally144/96 with the selected N512 sharding).
_LOCAL_NARROW_READER = _specialize(_NARROW_GDN_COMBINE_READER, {516: 2, 320: 2, 512: 1, 192: 1})
_LOCAL_NARROW_COMPUTE = _specialize(_NARROW_GDN_COMBINE_COMPUTE, {516: 1, 320: 1, 512: 1})


def local_narrow_gdn_combine(primary, residual, *, output=None, cores=None):
    assert len(primary.shape) == len(residual.shape) == 3
    batch, seq, pwidth = tuple(primary.shape)
    rb, rs, rwidth = tuple(residual.shape)
    assert (batch, seq) == (rb, rs) and pwidth >= 4160 and rwidth >= 2624
    assert tuple(primary.padded_shape)[:2] == tuple(residual.padded_shape)[:2]
    for t in (primary, residual):
        assert t.dtype == ttnn.float32 and t.layout == ttnn.TILE_LAYOUT
        mem = t.memory_config()
        assert mem == ttnn.DRAM_MEMORY_CONFIG or (
            mem.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED and mem.buffer_type == ttnn.BufferType.L1
        )
    if output is None:
        output = ttnn.empty(
            (batch, seq, 4160),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=primary.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    assert (
        tuple(output.shape) == (batch, seq, 4160) and tuple(output.padded_shape)[:2] == tuple(primary.padded_shape)[:2]
    )
    assert (
        output.dtype == ttnn.bfloat16
        and output.layout == ttnn.TILE_LAYOUT
        and output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    )
    assert output.buffer_address() not in (primary.buffer_address(), residual.buffer_address())
    total = batch * (primary.padded_shape[-2] // 32) * 130
    grid = primary.device().compute_with_storage_grid_size()
    workers = min(total, cores or grid.x * grid.y)
    assert 1 <= workers <= grid.x * grid.y
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    start = 0
    for i, c in enumerate(coords):
        count = total // workers + (i < total % workers)
        rr[c.x][c.y] = [
            primary.buffer_address(),
            residual.buffer_address(),
            primary.padded_shape[-1] // 32,
            residual.padded_shape[-1] // 32,
            start,
            count,
        ]
        wr[c.x][c.y] = [output.buffer_address(), start, count]
        cr[c.x][c.y] = [start, count]
        start += count
    rct = []
    for t in (primary, residual):
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    def kernel(source, config, args, ct):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=args,
            config=config,
        )

    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for i in (0, 1, 3):
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    kernels = [
        kernel(_LOCAL_NARROW_READER, ttnn.ReaderConfigDescriptor(), rr, rct),
        kernel(
            _NARROW_GDN_COMBINE_WRITER,
            ttnn.WriterConfigDescriptor(),
            wr,
            ttnn.TensorAccessorArgs(output).get_compile_time_args(),
        ),
        kernel(_LOCAL_NARROW_COMPUTE, config, cr, []),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=size,
            core_ranges=ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=i,
                    data_format=ttnn.bfloat16 if i == 2 else ttnn.float32,
                    page_size=2048 if i == 2 else 4096,
                )
            ],
        )
        for i, size in ((0, 8192), (1, 8192), (2, 4096), (3, 4096))
    ]
    ttnn.generic_op([primary, residual, output], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return output


class NarrowMultichipDecoder(MultichipDecoder):
    candidate_overrides = {}
    setup_record = None

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, policy=None):
        policy = {**(policy or {}), **cls.candidate_overrides}
        enabled = bool(policy.get("narrow_gdn_correction", False))
        # Super setup leaves the complete correction available, already zeroed
        # over local Z, and prepares the selected production row-MMRS copies.
        self = super().from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            policy={**policy, "narrow_gdn_correction": False},
        )
        self.narrow_gdn_correction = enabled and self.kind == "linear_attention"
        self.policy["narrow_gdn_correction"] = self.narrow_gdn_correction
        self.policy["correction_l1"] = bool(policy.get("correction_l1", False))
        cls.setup_record = {
            "layer_kind": self.kind,
            "narrow_enabled": self.narrow_gdn_correction,
            "correction_l1": self.policy["correction_l1"],
        }
        if not self.narrow_gdn_correction:
            return self
        assert self.policy.get("narrow_gdn_fused_combine", True), "this candidate requires native fused combine"
        self.policy["narrow_gdn_fused_combine"] = True
        correction_key = self.weight_corrections["packed_gdn"]
        full = ttnn.to_memory_config(self.weights[correction_key], ttnn.DRAM_MEMORY_CONFIG)
        assert full.dtype == ttnn.bfloat8_b and full.layout == ttnn.TILE_LAYOUT
        assert full.shape[0] == 5120 and full.shape[1] >= 4160
        qkv = ttnn.slice(full, (0, 0), (5120, 2560), (1, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ba = ttnn.slice(full, (0, 4096), (5120, 4160), (1, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        narrow = ttnn.concat([qkv, ba], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        assert tuple(narrow.shape) == (5120, 2624)
        verified = False
        if self.policy.get("narrow_gdn_verify_setup", True):
            import torch

            for source, actual in zip(ttnn.get_device_tensors(full), ttnn.get_device_tensors(narrow)):
                host = ttnn.to_torch(source)
                expected = torch.cat([host[:, :2560], host[:, 4096:4160]], -1)
                assert torch.count_nonzero(host[:, 2560:4096]).item() == 0
                assert torch.equal(ttnn.to_torch(actual), expected)
            verified = True
        # Reprepare ONLY the replaced matrix. Running the whole dictionary again
        # would replace other projection_info.logical_n values with already-
        # padded physical widths (e.g. packed_gdn4160 ->4608).
        weights = self.weights
        old_info = dict(self.projection_info)
        row_copies = {key: value.buffer_address() for key, value in self.row_parallel_weights.items()}
        self.weights = {correction_key: narrow}
        try:
            self._prepare_matmuls()
            weights[correction_key] = self.weights[correction_key]
        finally:
            self.weights = weights
        for key, info in old_info.items():
            if key != correction_key:
                assert self.projection_info[key] is info
        assert row_copies == {key: value.buffer_address() for key, value in self.row_parallel_weights.items()}
        cls.setup_record.update(
            {
                "primary_logical_width": 4160,
                "correction_logical_width": 2624,
                "correction_before_shape": list(full.shape),
                "correction_after_shape": list(self.weights[correction_key].shape),
                "correction_projection_logical_n": self.projection_info.get(correction_key, {}).get("logical_n"),
                "setup_qkv_ba_exact_and_z_zero": verified,
                "other_projection_metadata_preserved": True,
                "row_parallel_copies_preserved": True,
            }
        )
        print("LOCAL_NARROW_SETUP", json.dumps(cls.setup_record), flush=True)
        return self

    def _combine_correction(self, primary, correction, key):
        if self.narrow_gdn_correction and key == "packed_gdn":
            return local_narrow_gdn_combine(
                primary,
                correction,
                cores=self.policy.get("narrow_gdn_combine_cores"),
            )
        return super()._combine_correction(primary, correction, key)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--narrow", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--l1", action=argparse.BooleanOptionalAction, default=None)
    candidate, remaining = parser.parse_known_args()
    inspection = argparse.ArgumentParser(add_help=False)
    inspection.add_argument("--output", type=Path, required=True)
    inspection.add_argument("--baseline", action="store_true")
    destination, _ = inspection.parse_known_args(remaining)
    assert not destination.baseline, "produce single-chip references with the original runner"
    NarrowMultichipDecoder.candidate_overrides = {
        key: value
        for key, value in (
            ("narrow_gdn_correction", candidate.narrow),
            ("correction_l1", candidate.l1),
        )
        if value is not None
    }
    source = Path(__file__).resolve()
    command = shlex.join(
        [sys.executable, "-m", "models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_narrow", *sys.argv[1:]]
    )
    previous_mtime = destination.output.stat().st_mtime_ns if destination.output.exists() else None
    original_argv = sys.argv
    failure = None
    try:
        sys.argv = [original_argv[0], *remaining]
        with patch.object(runner, "MultichipDecoder", NarrowMultichipDecoder):
            runner.main()
    except BaseException:
        failure = traceback.format_exc()
        raise
    finally:
        sys.argv = original_argv
        produced = destination.output.exists() and destination.output.stat().st_mtime_ns != previous_mtime
        report = json.loads(destination.output.read_text()) if produced else {"pass": False}
        report["candidate"] = {
            "name": "local_narrow_gdn_correction",
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "command": command,
            "flag_overrides": NarrowMultichipDecoder.candidate_overrides,
            "setup": NarrowMultichipDecoder.setup_record,
            "reader_sha256": hashlib.sha256(_LOCAL_NARROW_READER.encode()).hexdigest(),
            "compute_sha256": hashlib.sha256(_LOCAL_NARROW_COMPUTE.encode()).hexdigest(),
            "writer_sha256": hashlib.sha256(_NARROW_GDN_COMBINE_WRITER.encode()).hexdigest(),
            "production_source_unchanged_by_candidate": True,
        }
        if failure is not None:
            report["candidate"]["exception"] = failure
            report["pass"] = False
        destination.output.parent.mkdir(parents=True, exist_ok=True)
        destination.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({"candidate_artifact": str(destination.output), "pass": report.get("pass", False)}), flush=True
        )


if __name__ == "__main__":
    main()
