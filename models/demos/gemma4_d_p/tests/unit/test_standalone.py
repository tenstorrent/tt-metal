# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks for the standalone Galaxy prefill boundary."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4_d_p.config import GALAXY_MESH_SHAPES, MeshConfig, ModeConfig
from models.demos.gemma4_d_p.tt.common import create_tt_model
from models.demos.gemma4_d_p.tt.model import _cp_chunk_major_row_order


@pytest.mark.parametrize("shape", GALAXY_MESH_SHAPES)
def test_galaxy_parallelism_uses_all_rows_and_columns(shape):
    config = MeshConfig(shape)
    assert config.prefill.sp == shape[0]
    assert config.prefill.tp == shape[1]
    assert config.total_devices == 32


@pytest.mark.parametrize("shape", [(1, 1), (1, 2), (2, 4), (1, 8), (1, 32), (8, 8)])
def test_smaller_or_multiple_galaxies_are_rejected(shape, expect_error):
    with expect_error(ValueError, "requires a Galaxy mesh"):
        MeshConfig(shape)


def test_disabling_cp_is_rejected(expect_error):
    with expect_error(ValueError, "must use all Galaxy rows"):
        MeshConfig((8, 4), prefill=ModeConfig(tp=4, sp=1))


@pytest.mark.parametrize("chunk_size", [0, -8192, 4096, 8193])
def test_invalid_chunk_geometry_fails_before_weight_loading(chunk_size, expect_error):
    with expect_error(ValueError, "positive|whole CP-local tiles|sliding window"):
        create_tt_model(SimpleNamespace(shape=(8, 4)), max_seq_len=32768, prefill_chunk_size=chunk_size)


@pytest.mark.parametrize("cp,chunk_size", [(8, 8192), (4, 4096), (8, 16384), (8, 32768)])
def test_rope_shards_follow_chunk_positions(cp, chunk_size):
    max_seq_len = 32768
    order = _cp_chunk_major_row_order(max_seq_len, cp, chunk_size).reshape(cp, -1)
    local_chunk = chunk_size // cp
    for rank in range(cp):
        for chunk in range(max_seq_len // chunk_size):
            start = chunk * chunk_size + rank * local_chunk
            torch.testing.assert_close(
                order[rank, chunk * local_chunk : (chunk + 1) * local_chunk],
                torch.arange(start, start + local_chunk),
            )


def test_imports_do_not_depend_on_original_gemma4():
    repo = Path(__file__).resolve().parents[5]
    script = """
import importlib
import pathlib
import sys

class BlockOriginalGemma4:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'models.demos.gemma4' or fullname.startswith('models.demos.gemma4.'):
            raise ImportError('Unexpected dependency on original Gemma4: ' + fullname)

sys.meta_path.insert(0, BlockOriginalGemma4())
for path in pathlib.Path('models/demos/gemma4_d_p').rglob('*.py'):
    if 'tests' in path.parts:
        continue
    name = '.'.join(path.with_suffix('').parts).removesuffix('.__init__')
    importlib.import_module(name)
"""
    subprocess.run([sys.executable, "-c", script], cwd=repo, check=True, capture_output=True, text=True)


def test_31b_config_defaults_match_supported_architecture():
    from models.demos.gemma4_d_p.tt.model_config import Gemma4ModelArgs

    args = Gemma4ModelArgs.from_hf_config(Gemma4ModelArgs())
    assert (args.hidden_size, args.intermediate_size, args.num_hidden_layers) == (5376, 21504, 60)
    assert (args.num_attention_heads, args.num_key_value_heads, args.num_global_key_value_heads) == (32, 16, 4)


@pytest.mark.parametrize(
    "field,value",
    [
        ("enable_moe_block", True),
        ("hidden_size_per_layer_input", 256),
        ("num_kv_shared_layers", 20),
        ("use_double_wide_mlp", True),
        ("hidden_size", 2816),
        ("num_hidden_layers", 30),
    ],
)
def test_non_31b_architectures_are_rejected(field, value, expect_error):
    from models.demos.gemma4_d_p.tt.model_config import Gemma4ModelArgs

    config = Gemma4ModelArgs()
    setattr(config, field, value)
    with expect_error(ValueError, "Only Gemma4-31B-it is supported"):
        Gemma4ModelArgs.from_hf_config(config)
