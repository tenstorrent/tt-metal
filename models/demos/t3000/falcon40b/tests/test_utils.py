# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# The reference loader lives in reference/load_falcon_weights.py so non-test code (the demo)
# can use it without importing from tests/. Re-exported here for the existing test imports.
from models.demos.t3000.falcon40b.reference.load_falcon_weights import load_falcon_reference_model  # noqa: F401


def load_hf_model(model_version):
    hugging_face_reference_model = load_falcon_reference_model(model_version)
    state_dict = hugging_face_reference_model.state_dict()

    return hugging_face_reference_model, state_dict


PREFILL_MODEL_CONFIGS = ("BFLOAT8_B-DRAM", "BFLOAT16-DRAM")
DECODE_MODEL_CONFIGS = ("BFLOAT8_B-SHARDED", "BFLOAT16-SHARDED")


def unsupported_mode_reason(
    llm_mode,
    num_devices,
    model_config_str=None,
    data_type=None,
    memcfg=None,
    decode_model_configs=DECODE_MODEL_CONFIGS,
    **_,
):
    """Return why a test case's llm_mode / model config / num_devices can't run, or None if it can.

    Pass it to @pytest.mark.skip_params (see conftest.py), which evaluates it at collection time. Takes either
    model_config_str or the data_type and memcfg it is built from.
    """
    if model_config_str is None:
        model_config_str = f"{data_type}-{memcfg}"
    if llm_mode == "prefill" and (model_config_str not in PREFILL_MODEL_CONFIGS or num_devices != 8):
        return "Prefill is only supported for DRAM memory config and 8 chips!"
    if llm_mode == "decode" and model_config_str not in decode_model_configs:
        return "Decode is only supported for SHARDED memory config!"
    return None
