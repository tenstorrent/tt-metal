# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
from _pytest.config.argparsing import Parser

from models.ttt_compat.demo.conftest import pytest_addoption


def _parse(args):
    parser = Parser(_ispytest=True)
    pytest_addoption(parser)
    return parser.parse(args)


@pytest.mark.parametrize("option", ["paged_attention", "token_accuracy", "stress_test"])
@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on", "false", "False", "0", "no", "off"])
def test_demo_boolean_values(option, value):
    options = _parse([f"--{option}", value])
    assert getattr(options, option) is (value.lower() in ("true", "1", "yes", "on"))


@pytest.mark.parametrize("option", ["paged_attention", "token_accuracy", "stress_test"])
def test_demo_boolean_unspecified_and_invalid(option, expect_error):
    assert getattr(_parse([]), option) is None
    with expect_error(pytest.UsageError, "expected a boolean"):
        _parse([f"--{option}", "typo"])


@pytest.mark.parametrize("block_size, blocks", [(32, 2048), (64, 4096)])
def test_page_params_json(block_size, blocks):
    params = {"page_block_size": block_size, "page_max_num_blocks_per_dp": blocks}
    assert _parse(["--page_params", json.dumps(params)]).page_params == params
    assert _parse([]).page_params is None


@pytest.mark.parametrize("value", ["not-json", "[]", "null", "1", '"string"'])
def test_page_params_rejects_invalid_objects(value, expect_error):
    with expect_error(pytest.UsageError, "--page_params"):
        _parse(["--page_params", value])


@pytest.mark.parametrize("disabled_mode", [None, "token_accuracy", "stress_test"])
@pytest.mark.parametrize(
    "page_params, expected_page_params",
    [
        (
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 2048},
            {"page_block_size": 32, "page_max_num_blocks_per_dp": 2048},
        ),
        ({"page_block_size": 64}, {"page_block_size": 64, "page_max_num_blocks_per_dp": 1024}),
        ({"page_max_num_blocks_per_dp": 2048}, {"page_block_size": 32, "page_max_num_blocks_per_dp": 2048}),
    ],
    ids=["full-page-params", "block-size-only", "block-count-only"],
)
@pytest.mark.parametrize(
    "cli_value, preset, expected",
    [("False", True, False), ("True", False, True), (None, True, True), (None, False, False)],
)
def test_paged_attention_override_reaches_model_builder(
    cli_value, preset, expected, disabled_mode, page_params, expected_page_params, monkeypatch, expect_error
):
    from models.ttt_compat.demo import simple_text_demo as demo

    argv = ["--page_params", json.dumps(page_params)]
    if cli_value is not None:
        argv += ["--paged_attention", cli_value]
    if disabled_mode is not None:
        argv += [f"--{disabled_mode}", "False"]
    options = _parse(argv)
    request = SimpleNamespace(
        config=SimpleNamespace(getoption=lambda name: getattr(options, name.removeprefix("--"))),
        node=SimpleNamespace(callspec=SimpleNamespace(id="cli-regression")),
    )

    class ReachedModelBuilder(Exception):
        pass

    def capture(**kwargs):
        assert kwargs["paged_attention"] is expected
        assert kwargs["page_params"] == expected_page_params
        raise ReachedModelBuilder

    def unexpected_skip(reason):
        pytest.fail(f"The demo did not apply the boolean override: {reason}")

    monkeypatch.setattr(demo.pytest, "skip", unexpected_skip)
    monkeypatch.setenv("MESH_DEVICE", "P150")
    monkeypatch.setenv("HF_MODEL", "")
    monkeypatch.setattr(demo, "hf_max_context_len", lambda: None)
    monkeypatch.setattr(demo, "prepare_generator_args", capture)
    with expect_error(ReachedModelBuilder, ""):
        demo.test_demo_text(
            input_prompts=["hello"],
            instruct=False,
            repeat_batches=1,
            max_seq_len=128,
            batch_size=1,
            max_generated_tokens=1,
            paged_attention=preset,
            page_params={"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},
            sampling_params={},
            optimizations=None,
            stop_at_eos=True,
            mesh_device=None,
            is_ci_env=False,
            is_ci_v2_env=False,
            ci_only=False,
            data_parallel=1,
            reset_seeds=None,
            request=request,
            token_accuracy=disabled_mode is not None,
            stress_test=disabled_mode is not None,
            enable_trace=False,
            model_location_generator=None,
            num_layers=1,
            mode="full",
            use_prefetcher=False,
        )
