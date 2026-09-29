# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Model, weights and reference output for the traced runner.

Split from performant_runner.py along the same line as the other demos: this
file owns everything that is true of the model regardless of how it is
dispatched, and the runner owns the trace and command-queue plumbing.

ModernBERT needs less of this than the encoders it is modelled on. The attention
masks and the two RoPE caches are built once in the model constructor and stay
resident, so the only thing that crosses from host to device per inference is
`input_ids` - a few KB. That is why the runner has a single staging buffer where
sentence_bert has five.
"""

import torch
from loguru import logger

import ttnn
from models.experimental.modernbert.common import build_inputs, load_config, load_torch_model
from models.experimental.modernbert.reference.modernbert import ModernBertModel
from models.experimental.modernbert.tests.pcc_utils import pcc
from models.experimental.modernbert.tt.modernbert_model import TtnnModernBertModel
from models.experimental.modernbert.tt.weights import deallocate_weights, prepare_weights

# The gate the rest of the suite uses (tests/test_ttnn_model.py). This checks the
# model, not the trace: the traced and non-traced paths run identical kernels at
# identical addresses, so the trace is checked against the non-traced output
# directly in tests/test_modernbert_performant.py.
VALID_PCC = 0.99


class ModernBertPerformanceRunnerInfra:
    def __init__(self, device, batch_size=1, seq_len=256, input_ids=None):
        self.device = device
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.config = load_config()
        hf = load_torch_model()
        reference = ModernBertModel(self.config)
        reference.load_state_dict(hf.state_dict(), strict=True)
        reference.eval()

        ids, attention_mask = build_inputs(seq_len=seq_len, batch_size=batch_size)
        if input_ids is not None and tuple(input_ids.shape) != tuple(ids.shape):
            # attention_mask below comes from build_inputs, so a caller-supplied
            # input_ids of another shape would silently be paired with the wrong mask.
            raise ValueError(f"input_ids must be {tuple(ids.shape)} for batch {batch_size} seq {seq_len}")
        self.input_ids = ids if input_ids is None else input_ids
        self.attention_mask = attention_mask

        # Against HuggingFace in fp32, matching tests/test_ttnn_model.py, so the
        # number here is comparable to the PCCs quoted in the README.
        with torch.no_grad():
            self.torch_output = hf(input_ids=self.input_ids, attention_mask=attention_mask).last_hidden_state

        self.parameters = prepare_weights(reference, device)
        self.model = TtnnModernBertModel(self.parameters, self.config, device, seq_len, attention_mask=attention_mask)
        self.input_tensor = None
        self.output_tensor = None

    def setup_inputs(self, input_ids=None):
        """Host-side input_ids, laid out the way the model consumes them.

        Returned on host rather than on device: the runner owns the device-side
        buffers, so their addresses stay fixed for the life of the trace.
        """
        ids = self.input_ids if input_ids is None else input_ids
        return ttnn.from_torch(ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def run(self):
        self.output_tensor = self.model(self.input_tensor)

    def validate(self, output_tensor=None):
        output = self.output_tensor if output_tensor is None else output_tensor
        got = ttnn.to_torch(output).reshape(self.torch_output.shape)
        p = pcc(self.torch_output, got.float())
        logger.info(f"ModernBERT batch={self.batch_size} seq={self.seq_len} PCC={p:.8f}")
        assert p >= VALID_PCC, f"traced model PCC {p:.8f} < {VALID_PCC}"
        return p

    def dealloc_output(self):
        if self.output_tensor is not None:
            ttnn.deallocate(self.output_tensor)
            self.output_tensor = None

    def release(self):
        self.model.deallocate()
        deallocate_weights(self.parameters)
