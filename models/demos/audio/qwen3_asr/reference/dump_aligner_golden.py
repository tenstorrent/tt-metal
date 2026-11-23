# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Golden outputs of Qwen3-ForcedAligner-0.6B (qwen-asr package, CPU fp32) for the TT aligner test.

For each <name>.wav with its <name>.txt transcript, saves <out>/<name>.pt holding the aligner
input ids, mel features, the time-bin logits at the <timestamp> positions and the word
timestamps the package returns. Needs `pip install qwen-asr`; does not need ttnn.

  python dump_aligner_golden.py --data /data/interleave --out /data/aligner_golden
"""
import argparse
import glob
import os

import soundfile as sf
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir of <name>.wav + <name>.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default="English")
    args = ap.parse_args()

    from qwen_asr import Qwen3ForcedAligner

    aligner = Qwen3ForcedAligner.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B", device_map="cpu", dtype=torch.float32)
    ts_id = aligner.timestamp_token_id
    os.makedirs(args.out, exist_ok=True)
    for wav_path in sorted(glob.glob(os.path.join(args.data, "*.wav"))):
        name = os.path.splitext(os.path.basename(wav_path))[0]
        text = open(os.path.join(args.data, name + ".txt")).read().strip()
        wav, sr = sf.read(wav_path, dtype="float32")
        words, input_text = aligner.aligner_processor.encode_timestamp(text, args.language)
        inputs = aligner.processor(text=[input_text], audio=[wav], return_tensors="pt", padding=True)
        with torch.inference_mode():
            logits = aligner.model.thinker(**inputs).logits[0]
        ids = inputs["input_ids"][0]
        (result,) = aligner.align(audio=(wav, sr), text=text, language=args.language)
        torch.save(
            {
                "input_ids": ids,
                "input_features": inputs["input_features"][0],
                "feature_attention_mask": inputs["feature_attention_mask"][0],
                "timestamp_logits": logits[ids == ts_id].float(),
                "words": words,
                "word_times": [(it.start_time, it.end_time) for it in result],
                "timestamp_token_id": ts_id,
                "timestamp_segment_time": aligner.timestamp_segment_time,
            },
            os.path.join(args.out, name + ".pt"),
        )
        print(f"{name}: {len(words)} words, {int((ids == ts_id).sum())} timestamps, {ids.numel()} tokens")


if __name__ == "__main__":
    main()
