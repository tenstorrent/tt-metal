# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The LLK 2.0 tt-metal kernel kernels/sdpa.cpp, the file itself, on one core: out = softmax(scale Q K^T) V.

The driver (llk2_sdpa_compute_api_test.cpp) stands in for tt-metal's build and for the reader and writer kernels. Q is
q_chunks chunks of sq tile rows; the keys come in k_chunks chunks of sk tiles, the last one last_width wide (the
others' padding is zero, as the reader leaves it). Every chunk is sent once per Q chunk, or with keep_kv once for all
the Q chunks (the reader's protocol for a single K chunk: the kernel keeps K and V over the Q chunks of a head).

Each case is a shape the kernel treats differently:
  blocks      a K chunk as wide as a Dest section (8 tiles), wider (16: two blocks) or narrower (4)
  K chunks    one (the first chunk is the last; also with a width that is not a whole block), two, three (the
              statistics go A -> B -> A)
  groups      1, 2 or 3 Q rows per group, one or several groups per Q chunk, one or two Q chunks
  K / V       sent with every Q chunk, or once and kept for both
  head        1 or 2 tiles (U goes through Dest two rows or one at a time)

test_llk2_sdpa_features adds what the op can ask for beyond the shape, one case each:
  mask        a mask tensor (the reader's tiles: a tile per tile of S, -inf where K is padding), on K chunks of one
              block and of a block and a rest; a partial last key tile (the writer's two tiles)
  V           narrower than Q and K (head_dim_v != head_dim), and wider than a Dest section (8 tiles): 12 and 16
  alpha       its exact exp (the op's exp_approx_mode off)
As above, the pack and unpack threads' 16 KB keep these to shapes with one block width per op: a wide V with two K
chunks does not fit.

Not here: several K chunks with a narrower last one. The kernel then has its block ops at two widths, and the pack
thread's code no longer fits the harness's 16 KB region for it (tt-metal's kernels have 70 KB). tt-metal's SDPA tests
cover those shapes (tests/nightly/blackhole/sdpa, TT_SDPA_LLK2=1).

The golden is the plain softmax in float32. The kernel's fast exp is an approximation, so the result is held to a
PCC and an absolute bound, as tt-metal's SDPA tests hold the 1.0 kernel.
"""

import math
import struct
from dataclasses import dataclass

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_SYNC, MATH_FIDELITY, TemplateParameter
from helpers.tilize_untilize import tilize_block, untilize_block
from llk2_common import TILE_DATUMS, TILE_DIM, unwrap_case
from llk2_paths import cpp_source, kernel_test_include_dirs


@dataclass
class LLK2_SDPA(TemplateParameter):
    sq: int  # Q tile rows per Q chunk
    sk: int  # key tiles per K chunk
    k_chunks: int
    last_width: int  # key tiles of the last K chunk
    dht: int  # head dimension in tiles
    group_rows: int
    q_chunks: int
    keep_kv: bool  # K and V sent once for all the Q chunks
    scale: float
    vdht: int  # head dimension of V and of the output, in tiles
    mask: int  # NO_MASK, PARTIAL_KEY_TILE or MASK_TENSOR
    partial_key_cols: int  # PARTIAL_KEY_TILE: keys in the last key tile
    exact_alpha_exp: bool

    def convert_to_cpp(self) -> str:
        bits = struct.unpack("<I", struct.pack("<f", self.scale))[0]
        return (
            f"constexpr std::uint32_t SQ = {self.sq};\n"
            f"constexpr std::uint32_t SK = {self.sk};\n"
            f"constexpr std::uint32_t K_CHUNKS = {self.k_chunks};\n"
            f"constexpr std::uint32_t LAST_WIDTH = {self.last_width};\n"
            f"constexpr std::uint32_t DHT = {self.dht};\n"
            f"constexpr std::uint32_t GROUP_ROWS = {self.group_rows};\n"
            f"constexpr std::uint32_t Q_CHUNKS = {self.q_chunks};\n"
            f"constexpr bool KEEP_KV = {str(self.keep_kv).lower()};\n"
            f"constexpr std::uint32_t SCALE_FP32 = {hex(bits)};\n"
            f"constexpr std::uint32_t VDHT = {self.vdht};\n"
            f"constexpr std::uint32_t MASK = {self.mask};\n"
            f"constexpr std::uint32_t PARTIAL_KEY_COLS = {self.partial_key_cols};\n"
            f"constexpr bool EXACT_ALPHA_EXP = {str(self.exact_alpha_exp).lower()};"
        )


# (sq, sk, k_chunks, last_width, dht, group_rows, q_chunks, keep_kv)
SHAPES = [
    (3, 8, 2, 8, 1, 3, 1, False),  # whole blocks, one group
    (2, 16, 2, 16, 1, 2, 1, False),  # two blocks per row
    (2, 4, 2, 4, 1, 2, 1, False),  # chunks narrower than a Dest section
    (3, 8, 1, 8, 1, 3, 1, False),  # a single K chunk
    (2, 8, 1, 5, 1, 2, 1, False),  # a single K chunk, narrow: not a whole block
    (4, 8, 3, 8, 1, 2, 2, False),  # three K chunks, two groups each, two Q chunks
    (3, 8, 2, 8, 2, 3, 1, False),  # head of two tiles
    (6, 8, 2, 8, 1, 3, 1, False),  # two groups of three rows
    (2, 8, 2, 8, 1, 1, 1, False),  # groups of one row
    (3, 8, 1, 8, 1, 3, 2, True),  # a single K chunk, kept for both Q chunks
    (
        2,
        1,
        2,
        1,
        1,
        2,
        1,
        False,
    ),  # K chunks of one tile: every run-time count has the bound 1
]

NO_MASK, PARTIAL_KEY_TILE, MASK_TENSOR = 0, 1, 2

# (shape as in SHAPES, vdht, mask, partial_key_cols, exact_alpha_exp)
FEATURES = [
    (
        (2, 8, 2, 8, 1, 2, 1, False),
        1,
        MASK_TENSOR,
        0,
        False,
    ),  # a mask tensor, a block per row
    (
        (2, 12, 1, 12, 1, 2, 1, False),
        1,
        MASK_TENSOR,
        0,
        False,
    ),  # ... a block of 8 tiles and one of 4
    (
        (2, 8, 1, 5, 1, 2, 1, False),
        1,
        MASK_TENSOR,
        0,
        False,
    ),  # ... a narrow K chunk: -inf tiles past it
    (
        (2, 8, 1, 5, 1, 2, 1, False),
        1,
        PARTIAL_KEY_TILE,
        13,
        False,
    ),  # 13 keys in the last key tile
    (
        (3, 8, 2, 8, 1, 3, 1, False),
        1,
        PARTIAL_KEY_TILE,
        20,
        False,
    ),  # ... of the second K chunk
    (
        (3, 8, 2, 8, 2, 3, 1, False),
        1,
        NO_MASK,
        0,
        False,
    ),  # V of one tile, Q and K of two
    (
        (2, 8, 1, 8, 1, 2, 1, False),
        12,
        NO_MASK,
        0,
        False,
    ),  # V of 12 tiles: blocks of 8 and 4 columns
    (
        (3, 8, 1, 8, 1, 3, 1, False),
        16,
        NO_MASK,
        0,
        False,
    ),  # V of 16 tiles: two blocks, groups of three rows
    ((3, 8, 2, 8, 1, 3, 1, False), 1, NO_MASK, 0, True),  # alpha's exact exp
]


def _tiles(x, rows_t, cols_t, fmt):
    """x [rows_t * 32, cols_t * 32] as a [rows_t, cols_t, 1024] grid of tilized tiles."""
    return tilize_block(
        x, dimensions=[rows_t * TILE_DIM, cols_t * TILE_DIM], stimuli_format=fmt
    ).view(rows_t, cols_t, TILE_DATUMS)


def _mask_stream(mask, q_chunks, sq, k_chunks, sk, keys_t, fmt):
    """A mask tensor [queries, keys_t * 32] as the reader sends it: per Q chunk and K chunk, sq rows of sk tiles,
    with -inf tiles where the K chunk is padding."""
    pad = k_chunks * sk - keys_t
    grid = torch.cat(
        [
            _tiles(mask, q_chunks * sq, keys_t, fmt),
            torch.full((q_chunks * sq, pad, TILE_DATUMS), float("-inf")),
        ],
        dim=1,
    )
    return torch.cat(
        [
            grid[qc * sq : (qc + 1) * sq, c * sk : (c + 1) * sk].reshape(-1)
            for qc in range(q_chunks)
            for c in range(k_chunks)
        ]
    )


def _made_masks(partial_key_cols, fmt):
    """The two tiles the writer makes for a partial last key tile: -inf, and 0 over the keys with -inf after them."""
    bar = torch.zeros(TILE_DIM, TILE_DIM)
    bar[:, partial_key_cols:] = float("-inf")
    tiles = [torch.full((TILE_DIM, TILE_DIM), float("-inf")), bar]
    return torch.cat(
        [
            tilize_block(
                t, dimensions=[TILE_DIM, TILE_DIM], stimuli_format=fmt
            ).flatten()
            for t in tiles
        ]
    )


def _run_sdpa(
    shape,
    formats,
    vdht=None,
    mask=NO_MASK,
    partial_key_cols=0,
    exact_alpha_exp=False,
    dest_acc=DestAccumulation.No,
):
    sq, sk, k_chunks, last_width, dht, group_rows, q_chunks, keep_kv = shape
    vdht = dht if vdht is None else vdht
    fmt = formats.input_format
    keys_t = (k_chunks - 1) * sk + last_width
    keys = keys_t * TILE_DIM - (
        TILE_DIM - partial_key_cols if mask == PARTIAL_KEY_TILE else 0
    )
    d = dht * TILE_DIM
    dv = vdht * TILE_DIM
    scale = 1.0 / math.sqrt(d)
    torch.manual_seed(sq * 1000 + sk * 100 + k_chunks * 10 + last_width)

    q = (torch.rand(q_chunks * sq * TILE_DIM, d) * 2 - 1).to(torch.bfloat16)
    k = (torch.rand(keys, d) * 2 - 1).to(torch.bfloat16)
    v = (torch.rand(keys, dv) * 2 - 1).to(torch.bfloat16)
    scores = q.float() @ k.float().T
    bias = None
    if mask == MASK_TENSOR:
        # The kernel adds the mask to the scores before the scale (the op divides its mask by the scale first). A
        # bias on every score, and blocks of keys hidden from blocks of queries.
        bias = (torch.rand(q_chunks * sq * TILE_DIM, keys) * 16 - 8).to(torch.bfloat16)
        hidden = torch.rand(q_chunks * sq * 2, keys_t * 2) < 0.3
        hidden = hidden.repeat_interleave(16, 0).repeat_interleave(16, 1)
        hidden[:, 0] = False  # every query sees a key
        bias = bias.float().masked_fill(hidden, float("-inf"))
        scores = scores + bias
    golden = torch.softmax(scale * scores, dim=-1) @ v.float()
    # the key tiles' padding: zeros, as the reader leaves it
    k = torch.cat([k, torch.zeros(keys_t * TILE_DIM - keys, d, dtype=k.dtype)])
    v = torch.cat([v, torch.zeros(keys_t * TILE_DIM - keys, dv, dtype=v.dtype)])

    # K and V as the reader sends them: per K chunk sk key tiles (the last one padded with zeros), K as a
    # [dht, sk] grid of its own tiles, V as [sk, dht]. Every chunk once per Q chunk, or once in all with keep_kv.
    pad = k_chunks * sk - keys_t
    k_grid = torch.cat(
        [_tiles(k, keys_t, dht, fmt), torch.zeros(pad, dht, TILE_DATUMS)]
    )
    v_grid = torch.cat(
        [_tiles(v, keys_t, vdht, fmt), torch.zeros(pad, vdht, TILE_DATUMS)]
    )
    k_chunks_flat = torch.cat(
        [
            k_grid[c * sk : (c + 1) * sk].permute(1, 0, 2).reshape(-1)
            for c in range(k_chunks)
        ]
    )
    v_chunks_flat = torch.cat(
        [v_grid[c * sk : (c + 1) * sk].reshape(-1) for c in range(k_chunks)]
    )
    # The reduce scaler as tt-metal's writer makes it: 1.0 in row 0 of every face, 0 elsewhere.
    unit = torch.zeros(TILE_DIM, TILE_DIM, dtype=torch.bfloat16)
    unit[0::16] = 1.0
    unit = tilize_block(
        unit, dimensions=[TILE_DIM, TILE_DIM], stimuli_format=fmt
    ).flatten()
    buffer_A = _tiles(q, q_chunks * sq, dht, fmt).flatten()
    sends = 1 if keep_kv else q_chunks
    masks = torch.zeros(0)
    if mask == MASK_TENSOR:
        masks = _mask_stream(bias, q_chunks, sq, k_chunks, sk, keys_t, fmt)
    elif mask == PARTIAL_KEY_TILE:
        masks = _made_masks(partial_key_cols, fmt)
    buffer_B = torch.cat(
        [
            k_chunks_flat.repeat(sends),
            v_chunks_flat.repeat(sends),
            unit.float(),
            masks.float(),
        ]
    )
    # S, P, L and 1 / L are Float32 with a 32-bit Dest: such a tile takes two of the scratch buffer's.
    wide = 2 if dest_acc == DestAccumulation.Yes else 1
    scratch_tiles = (
        wide * (3 * group_rows * sk + group_rows + 2 * sq)
        + 3 * group_rows
        + 2 * sq
        + 2 * sq * vdht
    )
    buffer_C = torch.zeros(scratch_tiles * TILE_DATUMS)
    out_tiles = q_chunks * sq * vdht

    configuration = TestConfig(
        cpp_source("llk2_sdpa_compute_api_test.cpp"),
        formats,
        templates=[
            MATH_FIDELITY(MathFidelity.HiFi2),  # the SDPA op's fidelity
            DEST_SYNC(),
            LLK2_SDPA(
                sq,
                sk,
                k_chunks,
                last_width,
                dht,
                group_rows,
                q_chunks,
                keep_kv,
                scale,
                vdht,
                mask,
                partial_key_cols,
                exact_alpha_exp,
            ),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            buffer_A,
            fmt,
            buffer_B,
            fmt,
            formats.output_format,
            tile_count_A=buffer_A.numel() // TILE_DATUMS,
            tile_count_B=buffer_B.numel() // TILE_DATUMS,
            tile_count_res=out_tiles,
            buffer_C=buffer_C,
            stimuli_C_format=fmt,
            tile_count_C=scratch_tiles,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
        include_dirs=kernel_test_include_dirs(),
    )

    res = configuration.run().result
    assert (
        len(res) == golden.numel()
    ), "Result tensor and golden tensor are not of the same length"
    out = untilize_block(
        torch.tensor(res, dtype=format_dict[formats.output_format]),
        stimuli_format=formats.output_format,
        dimensions=[q_chunks * sq * TILE_DIM, dv],
    ).float()
    pcc = torch.corrcoef(torch.stack([out.flatten(), golden.flatten()]))[0, 1].item()
    worst = (out - golden).abs().max().item()
    assert (
        pcc > 0.999 and worst < 0.02
    ), f"PCC {pcc:.6f}, largest error {worst:.4f} (values within 1)"


@parametrize(
    shape=SHAPES,
    formats=input_output_formats([DataFormat.Float16_b], same=True),
)
def test_llk2_sdpa(shape, formats):
    _run_sdpa(unwrap_case(shape), formats)


@parametrize(
    feature=FEATURES,
    formats=input_output_formats([DataFormat.Float16_b], same=True),
)
def test_llk2_sdpa_features(feature, formats):
    shape, vdht, mask, partial_key_cols, exact_alpha_exp = unwrap_case(feature)
    _run_sdpa(shape, formats, vdht, mask, partial_key_cols, exact_alpha_exp)


# With a 32-bit Dest the kernel's pack thread is larger, and for these it does not fit the harness's code region
# (by 12 to 668 bytes; tt-metal's is larger): two K chunks with a head of two tiles, with two groups, with a mask
# tensor, with a partial key tile in the second chunk. Each feature stays covered by a case that fits.
FP32_TOO_LARGE = [
    (3, 8, 2, 8, 2, 3, 1, False),
    (6, 8, 2, 8, 1, 3, 1, False),
    ((2, 8, 2, 8, 1, 2, 1, False), 1, MASK_TENSOR, 0, False),
    ((3, 8, 2, 8, 1, 3, 1, False), 1, PARTIAL_KEY_TILE, 20, False),
]


@parametrize(
    shape=[s for s in SHAPES if s not in FP32_TOO_LARGE],
    formats=input_output_formats([DataFormat.Float16_b], same=True),
)
def test_llk2_sdpa_fp32_dest(shape, formats):
    """The same shapes with a 32-bit Dest: blocks of 4 tiles, S, P, L and 1 / L in Float32."""
    _run_sdpa(unwrap_case(shape), formats, dest_acc=DestAccumulation.Yes)


@parametrize(
    feature=[f for f in FEATURES if f not in FP32_TOO_LARGE],
    formats=input_output_formats([DataFormat.Float16_b], same=True),
)
def test_llk2_sdpa_features_fp32_dest(feature, formats):
    shape, vdht, mask, partial_key_cols, exact_alpha_exp = unwrap_case(feature)
    _run_sdpa(
        shape,
        formats,
        vdht,
        mask,
        partial_key_cols,
        exact_alpha_exp,
        DestAccumulation.Yes,
    )
