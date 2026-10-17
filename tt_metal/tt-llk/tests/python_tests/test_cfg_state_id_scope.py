# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Build-only: ``cfg_state_id`` must be unambiguous from llk_lib scope.

The cfg-bank tracker is ``ckernel::cfg_state_id`` (declared in ``ckernel.h``, defined by
the firmware and by the harness's ``ckernel_helper.h``). ``ckernel_globals.h`` must not
declare a second, global-scope ``cfg_state_id``: every llk_lib header is a set of
global-scope functions after ``using namespace ckernel;``, and from there the unqualified
name would be ambiguous between the two declarations, the global one having no
definition anywhere. ``sources/cfg_state_id_scope_test.cpp`` reads the name from exactly
that scope; this test only builds it. Wormhole and Blackhole share the header text.
"""

from conftest import skip_for_quasar
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig


@skip_for_quasar
def test_cfg_state_id_resolves_from_llk_lib_scope():
    formats = input_output_formats([DataFormat.Float16_b], same=True)[0]
    input_dimensions = [32, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    configuration = TestConfig(
        "sources/cfg_state_id_scope_test.cpp",
        formats,
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=DestAccumulation.No,
    )
    # Build only: an ambiguous name fails the build here.
    configuration.prepare()

    elf_dir = (
        TestConfig.ARTEFACTS_DIR
        / configuration.test_name
        / configuration.variant_id
        / "elf"
    )
    missing = [
        n for n in ("unpack", "math", "pack") if not (elf_dir / f"{n}.elf").is_file()
    ]
    assert not missing, f"firmware ELFs were not built: {missing} (looked in {elf_dir})"
