# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


class YoloV4PipelineRunner:
    def __init__(self, test_infra):
        self._test_infra = test_infra

    def __call__(self, l1_input_tensor):
        self._test_infra.input_tensor = l1_input_tensor
        self._test_infra.run()
        return self._test_infra.output_tensor
