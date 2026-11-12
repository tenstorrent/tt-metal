# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
"""CPU FP32 reference: the public transformers ParakeetForTDT, used as the local correctness baseline.

This is the same implementation the evaluation oracle uses (AutoProcessor / generate), run on the
TT host CPU in FP32. It is the reference for the portable tests and the CPU baseline benchmark.
The evaluation oracle itself runs on an A100 and may differ slightly (see docs/PRECISION.md).
"""

import numpy as np


def strip_pad(row, pad):
    """Token row as a python list with trailing pad ids removed."""
    row = [int(t) for t in row]
    while row and row[-1] == pad:
        row.pop()
    return row


def row_nrmse(ref, out, lengths):
    """Per-row NRMSE over the valid frames: ||out - ref|| / ||ref||. Returns a list (one per row)."""
    ref, out = np.asarray(ref, np.float64), np.asarray(out, np.float64)
    vals = []
    for b, n in enumerate(lengths):
        r, o = ref[b, :n], out[b, :n]
        vals.append(float(np.linalg.norm(o - r) / max(np.linalg.norm(r), 1e-30)))
    return vals


class ParakeetReference:
    """FP32 CPU model wrapper with the backend's encode/transcribe signatures."""

    def __init__(self, weights_path, threads=0):
        import torch
        from transformers import ParakeetForTDT

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.model = ParakeetForTDT.from_pretrained(weights_path, dtype=torch.float32).eval()
        self.pad = self.model.generation_config.pad_token_id

    def _inputs(self, mel, mel_lengths):
        torch = self.torch
        mel = np.asarray(mel, dtype=np.float32)
        lens = np.asarray(mel_lengths, dtype=np.int64)
        mask = (np.arange(mel.shape[1])[None, :] < lens[:, None]).astype(np.int64)
        return torch.from_numpy(mel), torch.from_numpy(mask)

    def encode(self, mel, mel_lengths):
        x, mask = self._inputs(mel, mel_lengths)
        with self.torch.inference_mode():
            enc = self.model.encoder(input_features=x, attention_mask=mask).last_hidden_state
        return {"encoder": enc.float().numpy()}

    def transcribe(self, mel, mel_lengths):
        x, mask = self._inputs(mel, mel_lengths)
        with self.torch.inference_mode():
            out = self.model.generate(input_features=x, attention_mask=mask)
        seqs = out.sequences if hasattr(out, "sequences") else out
        return {"tokens": seqs.cpu().numpy().astype(np.int64)}
