# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-TTS Japanese benchmark runner.

Usage (GPU reference):
    python models/demos/qwen3_tts/evaluation/run_benchmark.py \
        --config models/demos/qwen3_tts/evaluation/benchmark_config.yaml \
        --backend reference \
        --output_dir results/

Usage (TT device):
    python models/demos/qwen3_tts/evaluation/run_benchmark.py \
        --config models/demos/qwen3_tts/evaluation/benchmark_config.yaml \
        --backend tt \
        --output_dir results/

Metrics computed:
    - CER: Character Error Rate via Whisper Large v3 ASR
    - UTMOS: Neural MOS prediction for naturalness
    - SIM: Speaker similarity (for voice cloning)
    - RTF: Real-Time Factor (wall-clock / audio duration)
    - PESQ: Perceptual quality (TT vs GPU waveform comparison)
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset(dataset_config):
    if dataset_config["type"] == "custom":
        with open(dataset_config["path"]) as f:
            data = json.load(f)
        samples = []
        for category in data["categories"]:
            for sample in category["samples"]:
                sample["category"] = category["name"]
                samples.append(sample)
        return samples
    elif dataset_config["type"] == "jsut":
        return load_jsut_dataset(dataset_config)
    raise ValueError(f"Unknown dataset type: {dataset_config['type']}")


def load_jsut_dataset(config):
    jsut_path = os.getenv("JSUT_DATA_PATH", config.get("path"))
    if not jsut_path or not Path(jsut_path).exists():
        print(f"JSUT dataset not found at {jsut_path}. Skipping.")
        return []

    samples = []
    transcript_file = Path(jsut_path) / "basic5000" / "transcript_utf8.txt"
    if transcript_file.exists():
        with open(transcript_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= config.get("subset_size", 100):
                    break
                parts = line.strip().split(":")
                if len(parts) >= 2:
                    utterance_id = parts[0]
                    text = ":".join(parts[1:])
                    wav_path = Path(jsut_path) / "basic5000" / "wav" / f"{utterance_id}.wav"
                    samples.append({
                        "id": utterance_id,
                        "text": text,
                        "category": "jsut",
                        "ref_audio": str(wav_path) if wav_path.exists() else None,
                    })
    return samples


def generate_all_reference(model, samples, gen_config, language="japanese"):
    """Generate speech for all samples using HF reference model."""
    from models.demos.qwen3_tts.reference.functional import generate_reference

    import torch

    results = []
    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {sample['id']}: {sample['text'][:40]}...")
        t0 = time.time()
        wavs, sr = generate_reference(
            model,
            text=sample["text"],
            language=language,
            max_new_tokens=gen_config.get("max_new_tokens", 2048),
            temperature=gen_config.get("temperature", 0.9),
            top_k=gen_config.get("top_k", 50),
            top_p=gen_config.get("top_p", 1.0),
            repetition_penalty=gen_config.get("repetition_penalty", 1.05),
        )
        elapsed = time.time() - t0

        wav = wavs[0].cpu().numpy() if isinstance(wavs[0], torch.Tensor) else wavs[0]
        results.append({
            "id": sample["id"],
            "text": sample["text"],
            "category": sample.get("category", "unknown"),
            "wav": wav,
            "sr": sr,
            "elapsed": elapsed,
            "duration": len(wav) / sr,
            "rtf": elapsed / (len(wav) / sr) if len(wav) > 0 else float("inf"),
        })
    return results


def generate_all_tt(generator, samples, gen_config, language="japanese"):
    """Generate speech for all samples using TT device pipeline."""
    results = []
    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {sample['id']}: {sample['text'][:40]}...")

        ref_audio = None
        ref_sr = 24000
        if sample.get("ref_audio"):
            import soundfile as sf

            ref_audio_raw, ref_sr = sf.read(sample["ref_audio"], dtype="float32")
            if ref_audio_raw.ndim > 1:
                ref_audio_raw = ref_audio_raw.mean(axis=1)
            ref_audio = ref_audio_raw

        t0 = time.time()
        waveform, sr = generator.generate(
            text=sample["text"],
            language=language,
            ref_audio=ref_audio,
            ref_sr=ref_sr,
            max_new_tokens=gen_config.get("max_new_tokens", 2048),
            temperature=gen_config.get("temperature", 0.9),
            top_k=gen_config.get("top_k", 50),
            top_p=gen_config.get("top_p", 1.0),
        )
        elapsed = time.time() - t0

        results.append({
            "id": sample["id"],
            "text": sample["text"],
            "category": sample.get("category", "unknown"),
            "wav": waveform,
            "sr": sr,
            "elapsed": elapsed,
            "duration": len(waveform) / sr,
            "rtf": elapsed / (len(waveform) / sr) if len(waveform) > 0 else float("inf"),
        })
    return results


def compute_cer(results, asr_config):
    """Compute CER using Whisper ASR."""
    try:
        from models.demos.qwen3_tts.evaluation.metrics.cer_eval import compute_cer_batch

        return compute_cer_batch(results, asr_config)
    except ImportError:
        print("CER evaluation not available (missing dependencies)")
        return {}


def compute_utmos(results, utmos_config):
    """Compute UTMOS naturalness scores."""
    try:
        from models.demos.qwen3_tts.evaluation.metrics.utmos_eval import compute_utmos_batch

        return compute_utmos_batch(results, utmos_config)
    except ImportError:
        print("UTMOS evaluation not available (missing dependencies)")
        return {}


def compute_pesq_scores(tt_results, ref_results):
    """Compute PESQ between TT and reference waveforms (requires paired samples)."""
    try:
        from pesq import pesq

        scores = {}
        ref_map = {r["id"]: r for r in ref_results}
        for r in tt_results:
            if r["id"] not in ref_map:
                continue
            ref_wav = ref_map[r["id"]]["wav"]
            tt_wav = r["wav"]
            min_len = min(len(ref_wav), len(tt_wav))
            if min_len < 16000:
                continue
            score = pesq(16000, ref_wav[:min_len], tt_wav[:min_len], "wb")
            scores[r["id"]] = float(score)
        return scores
    except ImportError:
        print("PESQ evaluation not available (pip install pesq)")
        return {}


def summarize_results(results, cer_scores, utmos_scores, pesq_scores=None):
    """Aggregate results by category and overall."""
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"rtf": [], "duration": []}
        categories[cat]["rtf"].append(r["rtf"])
        categories[cat]["duration"].append(r["duration"])

    summary = {
        "overall": {
            "num_samples": len(results),
            "total_audio_duration": sum(r["duration"] for r in results),
            "total_elapsed": sum(r["elapsed"] for r in results),
            "mean_rtf": float(np.mean([r["rtf"] for r in results])),
            "mean_cer": float(np.mean(list(cer_scores.values()))) if cer_scores else None,
            "mean_utmos": float(np.mean(list(utmos_scores.values()))) if utmos_scores else None,
            "mean_pesq": float(np.mean(list(pesq_scores.values()))) if pesq_scores else None,
        },
        "by_category": {},
    }

    for cat, data in categories.items():
        cat_ids = [r["id"] for r in results if r["category"] == cat]
        summary["by_category"][cat] = {
            "num_samples": len(data["rtf"]),
            "mean_rtf": float(np.mean(data["rtf"])),
            "mean_duration": float(np.mean(data["duration"])),
            "mean_cer": float(np.mean([cer_scores[sid] for sid in cat_ids if sid in cer_scores]))
            if cer_scores
            else None,
            "mean_utmos": float(np.mean([utmos_scores[sid] for sid in cat_ids if sid in utmos_scores]))
            if utmos_scores
            else None,
        }

    return summary


def check_pass_criteria(summary, criteria):
    """Check if results meet the pass criteria."""
    results = []
    if criteria.get("max_rtf") and summary["overall"]["mean_rtf"] is not None:
        passed = summary["overall"]["mean_rtf"] < criteria["max_rtf"]
        results.append(("RTF", passed, f"{summary['overall']['mean_rtf']:.3f} < {criteria['max_rtf']}"))

    if criteria.get("min_pesq") and summary["overall"]["mean_pesq"] is not None:
        passed = summary["overall"]["mean_pesq"] > criteria["min_pesq"]
        results.append(("PESQ", passed, f"{summary['overall']['mean_pesq']:.2f} > {criteria['min_pesq']}"))

    return results


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Japanese benchmark")
    parser.add_argument(
        "--config", type=str, default="models/demos/qwen3_tts/evaluation/benchmark_config.yaml"
    )
    parser.add_argument("--backend", type=str, choices=["reference", "tt"], default="reference")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--dataset", type=str, default="custom_ja", help="Dataset name from config (custom_ja, jsut)"
    )
    parser.add_argument("--save_audio", action="store_true", help="Save generated audio files")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument(
        "--compare_dir", type=str, default=None, help="Directory with reference results for PESQ comparison"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir or config["output"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    dataset_config = config["datasets"][args.dataset]
    samples = load_dataset(dataset_config)
    if not samples:
        print("No samples loaded. Exiting.")
        return

    if args.max_samples:
        samples = samples[: args.max_samples]

    print(f"Loaded {len(samples)} samples from '{args.dataset}'")

    if args.backend == "reference":
        import torch

        from models.demos.qwen3_tts.reference.functional import load_reference_model

        model = load_reference_model(
            config["model"]["path"],
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        print("Generating speech (reference)...")
        results = generate_all_reference(model, samples, config["generation"], config["model"]["language"])
    else:
        import ttnn

        from models.demos.qwen3_tts.tt.generator import TTSGenerator

        os.environ["HF_MODEL"] = config["model"]["path"]

        device_ids = ttnn.get_device_ids()
        mesh_device = ttnn.open_mesh_device(
            ttnn.MeshShape(1, len(device_ids)),
            dispatch_core_config=ttnn.DispatchCoreConfig(
                ttnn.DispatchCoreType.ETH
                if len(device_ids) > 1
                else ttnn.DispatchCoreType.WORKER
            ),
        )
        ttnn.enable_program_cache(mesh_device)

        print("Building TTS generator on TT device...")
        generator = TTSGenerator.build(
            config["model"]["path"],
            mesh_device,
            max_seq_len=config["generation"].get("max_new_tokens", 2048) + 512,
        )

        print("Generating speech (TT)...")
        results = generate_all_tt(generator, samples, config["generation"], config["model"]["language"])

        ttnn.close_mesh_device(mesh_device)

    if args.save_audio:
        import soundfile as sf

        audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        for r in results:
            sf.write(os.path.join(audio_dir, f"{r['id']}.wav"), r["wav"], r["sr"])
        print(f"Saved {len(results)} audio files to {audio_dir}")

    print("Computing metrics...")
    cer_scores = (
        compute_cer(results, config["metrics"]["cer"]) if config["metrics"]["cer"]["enabled"] else {}
    )
    utmos_scores = (
        compute_utmos(results, config["metrics"]["utmos"]) if config["metrics"]["utmos"]["enabled"] else {}
    )

    pesq_scores = {}
    if args.compare_dir and config["metrics"].get("pesq", {}).get("enabled"):
        ref_results_path = os.path.join(args.compare_dir, "audio")
        if Path(ref_results_path).exists():
            import soundfile as sf

            ref_results = []
            for r in results:
                ref_wav_path = os.path.join(ref_results_path, f"{r['id']}.wav")
                if Path(ref_wav_path).exists():
                    wav, sr = sf.read(ref_wav_path, dtype="float32")
                    ref_results.append({"id": r["id"], "wav": wav, "sr": sr})
            if ref_results:
                pesq_scores = compute_pesq_scores(results, ref_results)

    summary = summarize_results(results, cer_scores, utmos_scores, pesq_scores)
    summary["config"] = {
        "model": config["model"]["name"],
        "backend": args.backend,
        "dataset": args.dataset,
        "num_samples": len(samples),
    }

    output_file = os.path.join(output_dir, f"benchmark_{args.backend}_{args.dataset}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")

    print("\n=== Benchmark Summary ===")
    print(f"Backend: {args.backend}")
    print(f"Samples: {summary['overall']['num_samples']}")
    print(f"Total audio: {summary['overall']['total_audio_duration']:.1f}s")
    print(f"Mean RTF: {summary['overall']['mean_rtf']:.3f}")
    if summary["overall"]["mean_cer"] is not None:
        print(f"Mean CER: {summary['overall']['mean_cer']:.4f}")
    if summary["overall"]["mean_utmos"] is not None:
        print(f"Mean UTMOS: {summary['overall']['mean_utmos']:.3f}")
    if summary["overall"]["mean_pesq"] is not None:
        print(f"Mean PESQ: {summary['overall']['mean_pesq']:.2f}")

    print("\nBy category:")
    for cat, data in summary["by_category"].items():
        line = f"  {cat}: RTF={data['mean_rtf']:.3f}, dur={data['mean_duration']:.2f}s"
        if data["mean_cer"] is not None:
            line += f", CER={data['mean_cer']:.4f}"
        if data["mean_utmos"] is not None:
            line += f", UTMOS={data['mean_utmos']:.3f}"
        print(line)

    if "pass_criteria" in config:
        print("\n=== Pass Criteria ===")
        criteria_results = check_pass_criteria(summary, config["pass_criteria"])
        all_pass = True
        for name, passed, detail in criteria_results:
            status = "PASS" if passed else "FAIL"
            print(f"  {name}: {status} ({detail})")
            if not passed:
                all_pass = False
        if criteria_results:
            print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
