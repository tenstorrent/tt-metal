"""Census only, never committed: production datacopy configurations (dc-prod, dc_prod_cfg.json), ported for the floor audit onto the final
heads' test infra. The runner is the 11-21 census farm's (dcH23 test_eltwise_unary_datacopy.py: extra templates, dest sync, force SrcA, the
constant block count for production's pack_block<n>, compile-time formats, production's LTO flags); the source is the farm's
(eltwise_unary_datacopy_test.cpp, the head form on head arms, main's form on main arms)."""
import json
import os
from dataclasses import dataclass

import pytest
from helpers.data_format_inference import unpacks_to_dest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import BlocksCalculationAlgorithm, DestAccumulation, DestSync, Tilize
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.perf.core import ALL_PERF_RUN_TYPES, create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_variant_parameters import (
    DATACOPY_FLAG,
    DEST_INDEX,
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    TILIZE,
    UNPACK_BLOCK,
    TemplateParameter,
    generate_input_dim,
)

CFG = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dc_prod_cfg.json")))
WANT = os.environ.get("DC_CENSUS_IDS")
if WANT:
    _w = {l.split("[", 1)[1].rsplit("]", 1)[0] for l in open(WANT) if "[" in l}
    CFG = [c for c in CFG if c["id"] in _w]
DC_FID = os.environ.get("DC_CENSUS_FID", "1") == "1"


class DC_PACK_NT_T(DATACOPY_FLAG):
    """dc census 11-19: the block's tile count as a constant, for production's pack_block<n>."""

    def __init__(self, n: int):
        self.name = "DC_PACK_NT"
        self.n = n

    def convert_to_cpp(self) -> str:
        return f"#define DC_PACK_NT {self.n}"


@dataclass
class DC_PROD(TemplateParameter):
    dc_early: bool = False
    dc_full: bool = False

    def convert_to_cpp(self) -> str:
        # room-dc: the head pack (cpack_common.h) reads MATH_FIDELITY, which the JIT defines for every kernel; non-matmul kernels ignore its value
        fid = ["constexpr ckernel::MathFidelity MATH_FIDELITY = ckernel::MathFidelity::HiFi4;"] if DC_FID else []
        return "\n".join(fid + [d for d, on in (("#define DATACOPY_EARLY_POLL 1", self.dc_early), ("#define DATACOPY_SYNC_FULL 1", self.dc_full)) if on])


def _run_dc(formats, dest_acc, num_faces, input_dimensions, *, perf_report, extra_templates, dest_sync, force_srca, unpack_block):
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    unpack_to_dest = False if force_srca else unpacks_to_dest(formats.input_format, dest_acc, Tilize.No)
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync, dest_acc, formats, input_dimensions, TILE_DIMENSIONS, BlocksCalculationAlgorithm.Standard
    )
    pack_block = unpack_block and num_tiles_in_block > 1
    back_to_back = (
        unpack_block
        and num_faces == 4
        and not unpack_to_dest
        and not formats.input_format.is_block_float()
        and not formats.input_format.is_mx_format()
    )
    test_config_kwargs = {
        "test_name": "sources/eltwise_unary_datacopy_test.cpp",
        "formats": formats,
        "templates": [
            generate_input_dim(input_dimensions, input_dimensions),
            TILIZE(Tilize.No),
            *extra_templates,
            UNPACK_BLOCK(unpack_block),
            *[
                DATACOPY_FLAG(name)
                for name, on in (("DATACOPY_PACK_BLOCK", pack_block), ("DATACOPY_TILES_BACK_TO_BACK", back_to_back))
                if on
            ],
            DC_PACK_NT_T(num_tiles_in_block),
        ],
        "runtimes": [
            DEST_INDEX(0),
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            LOOP_FACTOR(32),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
        ),
        "dest_acc": dest_acc,
        "unpack_to_dest": unpack_to_dest,
        "compile_time_formats": True,
        "kernel_flags": ["-flto=1", "-flto-partition=one"],
    }
    create_test_or_perf_config(is_perf=True, run_types=ALL_PERF_RUN_TYPES, test_config_kwargs=test_config_kwargs).run(perf_report)


@pytest.mark.perf
@pytest.mark.parametrize("cfg", CFG, ids=[c["id"] for c in CFG])
def test_perf_dc_prod(perf_report, cfg):
    _run_dc(
        InputOutputFormat(DataFormat[cfg["fin"]], DataFormat[cfg["fout"]]),
        DestAccumulation[cfg["acc"]],
        cfg["nf"],
        cfg["dims"],
        perf_report=perf_report,
        extra_templates=(DC_PROD(dc_early=bool(cfg["early"]), dc_full=bool(cfg["full"])),),
        dest_sync=DestSync.Full if cfg["full"] else DestSync.Half,
        force_srca=bool(cfg.get("srca")),
        unpack_block=bool(cfg["block"]),
    )
