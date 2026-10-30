# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PREPARED_ALL (P.14, OV-23): the block MoE's fused tail (models/demos/moe_block/tt/tail_a2a.py TailA2A) with its
prepared program keyed on the tensors' shapes and memory configs, the y / se / out addresses patched per call.

TailA2A.run keys its cache on the (y, se, out) ADDRESSES, one 16-entry FIFO for all 40 layers (one instance a block):
any allocation change in the forward (a buffer kept between calls, DS41-146) shifts those addresses and every layer
rebuilds its descriptor and prepares it again (P.4: 22 -> 188 ms a chunk). Here one descriptor is built (at the first
call of a shape) and prepared with io [y, out, z, zrw (, se)]; ttnn.bringup.run_prepared then patches the runtime-arg
slots holding y / se / out (the workers' reader arg 0 / 1, writer arg 0, the port sender's arg 0). z and zrw are the
instance's own buffers. Every other runtime arg (semaphores, fabric connections, schedule) depends only on the mesh
and the shapes, so the device program, its arguments and the results are those of the address-keyed path.

Guard (prepared_program contract: a frozen constant equal to an io address at prepare is taken for it): the prepared
patch count must equal the address slots the descriptor holds, else the shape key is not used for that shape and the
call takes TailA2A's own address-keyed path (counted in ``fallbacks``).

moe_block/ is outside this model's paths, so the class is installed on the block's TailA2A instance
(``install(block)``) instead of changing tail_a2a.py; TailA2A's own run is the off path (PREPARED_ALL=off)."""

from __future__ import annotations

import ttnn
from models.demos.moe_block.tt.tail_a2a import TailA2A


def _spec(t):
    return None if t is None else (tuple(t.shape), t.dtype, t.layout, str(t.memory_config()))


class ShapeKeyedTailA2A(TailA2A):
    """TailA2A whose run() dispatches one prepared program per (y, se, out) shape / dtype / layout / memory config."""

    def _expected_patches(self, has_se: bool) -> int:
        # per chip: each worker reader y (+ se), writer out, z, zrw; each port reader z, zrw; each port sender out
        return sum(len(self.workers) * (4 + has_se) + 3 * len(ch["ports"]) for ch in self.chips.values())

    def run(self, y_rm, se, out=None):
        if out is None:
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, self.rows, self.H // self.G]),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                self.mesh,
                ttnn.DRAM_MEMORY_CONFIG,
            )
        shaped = self.__dict__.setdefault("shaped", {})
        key = (_spec(y_rm), _spec(se), _spec(out))
        pp = shaped.get(key, False)
        if pp is False:
            self.shape_misses = getattr(self, "shape_misses", 0) + 1
            io = [y_rm, out, self.z, self.zrw] + ([se] if se is not None else [])
            md = ttnn.MeshProgramDescriptor()
            for coord in self.chips:
                r, c = coord
                md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = self._program(
                    coord, y_rm, se, out
                )
            pp = ttnn.bringup.prepare_program(md, io)
            if pp.patch_count != self._expected_patches(se is not None):
                self.fallbacks = getattr(self, "fallbacks", 0) + 1
                pp = None  # (a frozen value matched an io address: TailA2A's address-keyed path for this shape)
            shaped[key] = pp
        if pp is None:
            return TailA2A.run(self, y_rm, se, out)
        # (looked up on ttnn.bringup at each call: the bring-up capture wraps run_prepared, F201)
        ttnn.bringup.run_prepared(pp, [y_rm, out, self.z, self.zrw] + ([se] if se is not None else []))
        return out


def install(block) -> None:
    """PREPARED_ALL=on: the MoEBlock's fused tail (FusedTail.a2a, MOE_TAIL=fused, no latent) runs ShapeKeyedTailA2A.
    Its address-keyed warm-up entries stay in TailA2A.cache (unused)."""
    tail = getattr(block, "tail", None)
    a2a = getattr(tail, "a2a", None)
    if a2a is not None and type(a2a) is TailA2A:
        a2a.__class__ = ShapeKeyedTailA2A
