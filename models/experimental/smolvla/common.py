# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os


def get_model_path(model_location_generator=None):
    if model_location_generator is None or "TT_GH_CI_INFRA" not in os.environ:
        return "lerobot/smolvla_base"
    else:
        return str(model_location_generator("vla-models/smolvla_base", model_subdir="", download_if_ci_v2=True))
