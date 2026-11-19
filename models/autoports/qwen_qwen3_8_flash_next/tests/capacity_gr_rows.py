# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test-only GR query splitting; original projection math stays unchanged."""


class SplitGRQueryRows:
    def __init__(self, original, name, rows=16):
        if rows != 16 or (original.input_width, original.output_width) not in ((10240, 320), (320, 10240)):
            raise ValueError("Only target GR down/up and fixed16-row split supported")
        self.original = original
        self.rows = rows
        self.report = {
            "name": name,
            "input_width": original.input_width,
            "output_width": original.output_width,
            "rows": rows,
            "calls_by_count": {},
            "scope": "Test-only BF16 row slices before original FP32 expansion; original K tree and BF16 boundary unchanged; concat dim2",
        }

    def __call__(self, value):
        import ttnn

        count = value.shape[-2]
        if tuple(value.shape) != (1, 1, count, self.original.input_width) or not 1 <= count <= 32:
            raise ValueError("Unexpected GR logical shape")
        if value.dtype != ttnn.bfloat16 or value.layout != ttnn.TILE_LAYOUT:
            raise ValueError("Splitting must happen on the original BF16 tiled input")
        key = str(count)
        self.report["calls_by_count"][key] = self.report["calls_by_count"].get(key, 0) + 1
        if count <= self.rows:
            return self.original(value)
        outputs = []
        for start in range(0, count, self.rows):
            part = value[:, :, start : min(start + self.rows, count), :]
            outputs.append(self.original(part))
            del part
        return ttnn.concat(outputs, dim=2)


def install_split_gr_down(model, rows=16):
    """Only two K10240 down projections have the large product allocation."""
    reports = []
    for name in ("attn_mix", "mlp_mix"):
        module = getattr(model, name)
        if module.precise_down.input_width != 10240 or module.precise_up.input_width != 320:
            raise ValueError("Unexpected target GR projection geometry")
        wrapper = SplitGRQueryRows(module.precise_down, name + ".precise_down", rows)
        module.precise_down = wrapper
        reports.append(wrapper.report)
    return {
        "enabled": True,
        "test_only": True,
        "policy": "Split only attn_mix.precise_down and mlp_mix.precise_down; precise_up unchanged",
        "projections": reports,
    }
