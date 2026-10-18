"""Shared prompt suite with exact pinned HF chat formatting and paired controls."""

import argparse
import json
import os
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, GenerationConfig


def canonicalize_eos(root, checkpoint):
    """Reuse deterministic greedy prefixes; preserve original termination evidence.

    EOS stopping changes sequence termination only, not logits or token choice
    before the first EOS. All controls used explicit do_sample=False.
    """
    import hashlib

    output = root / "doc/full_model"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    config = GenerationConfig.from_pretrained(checkpoint, local_files_only=True)
    eos = config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    preserved = output / "pre_eos_repair"
    preserved.mkdir(exist_ok=True)
    hashes = {}

    def preserve(path):
        relative = path.relative_to(output)
        dest = preserved / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(path.read_bytes())
        hashes[str(relative)] = hashlib.sha256(dest.read_bytes()).hexdigest()

    def trim(ids):
        return next((ids[: i + 1] for i, token in enumerate(ids) if token in eos), ids)

    hf_path, tt_path = output / "qualitative_hf.json", output / "qualitative_tt.json"
    preserve(hf_path)
    preserve(tt_path)
    hf = json.loads((preserved / hf_path.name).read_text())
    tt = json.loads((preserved / tt_path.name).read_text())
    assert hf["generation"]["do_sample"] is False
    changes = []
    for h, t in zip(hf["prompts"], tt, strict=True):
        assert h["id"] == t["id"]
        counts = {}
        for backend, row in (("hf", h), ("tt", t)):
            original = row["token_ids"]
            row["token_ids"] = trim(original)
            assert original[: len(row["token_ids"])] == row["token_ids"]
            row["completion"] = tokenizer.decode(row["token_ids"], skip_special_tokens=False)
            counts[backend] = dict(original=len(original), canonical=len(row["token_ids"]))
        dest = output / f"autoregressive_suite_{h['id']}"
        meta_path = dest / "autoregressive_meta.json"
        preserve(meta_path)
        metadata = json.loads(meta_path.read_text())
        for backend, row in (("hf", h), ("tt", t)):
            completion = dest / f"{backend}_completion.txt"
            preserve(completion)
            completion.write_text(row["completion"])
            metadata[backend] = dict(token_ids=row["token_ids"], num_tokens=len(row["token_ids"]))
        metadata["eos_prefix_provenance"] = "../eos_prefix_validation.json"
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        changes.append(dict(id=h["id"], counts=counts))
    hf["generation"]["eos_token_id"] = sorted(eos)
    hf["termination_provenance"] = "eos_prefix_validation.json"
    hf_path.write_text(json.dumps(hf, indent=2) + "\n")
    tt_path.write_text(json.dumps(tt, indent=2) + "\n")
    report = dict(
        status="pass",
        canonical_eos=sorted(eos),
        changes=changes,
        original_sha256=hashes,
        method="Exact original greedy prefixes sliced at the first canonical EOS; no generated token changed",
        reason="EOS lists affect stopping only. HF do_sample=False and TT semantic greedy are unchanged; "
        "new generator behavior is separately tested by scheduler_cpu.json and tracked hardware regression.",
        raw_results="pre_eos_repair; final01.json and its log also retain the original TT completions",
    )
    (output / "eos_prefix_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def run_tt(gen, root, *, output_dir=None):
    output = Path(output_dir) if output_dir is not None else root / "doc/full_model"
    path = root / "doc/full_model/qualitative_hf.json"
    controls = json.loads(path.read_text())
    results = []
    for item in controls["prompts"]:
        prompt = item["prompt_token_ids"]
        ids = gen.generate(prompt, 128, stop_on_eos=True)
        text = gen.tokenizer.decode(ids, skip_special_tokens=False)
        dest = output / f"autoregressive_suite_{item['id']}"
        dest.mkdir(exist_ok=True)
        (dest / "hf_completion.txt").write_text(item["completion"])
        (dest / "tt_completion.txt").write_text(text)
        (dest / "rendered_prompt.txt").write_text(item["rendered_prompt"])
        metadata = dict(
            hf_model_id="Qwen/Qwen3.8-Flash-Next",
            prompt_file=str(dest / "rendered_prompt.txt"),
            prompt_text=item["rendered_prompt"],
            prompt_token_ids=prompt,
            max_new_tokens=128,
            hf=dict(token_ids=item["token_ids"], num_tokens=len(item["token_ids"])),
            tt=dict(token_ids=ids, num_tokens=len(ids)),
            prompt_mode="chat",
            hf_control_source=str(path),
        )
        (dest / "autoregressive_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
        results.append(dict(id=item["id"], completion=text, token_ids=ids, perf=dict(gen.last_perf)))
        print("QUALITATIVE_TT", item["id"], repr(text), flush=True)
        (output / "qualitative_tt.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "tt", "canonicalize-eos"], required=True)
    ap.add_argument("--checkpoint", required=True)
    a = ap.parse_args()
    torch.set_num_threads(8)
    root = Path("models/autoports/qwen_qwen3_8_flash_next")
    if a.backend == "canonicalize-eos":
        canonicalize_eos(root, a.checkpoint)
        return
    if a.backend == "tt":
        import ttnn
        from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator

        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=768000000)
        gen = None
        try:
            gen = build_generator(root, mesh, checkpoint=a.checkpoint)
            gen.prepare()
            run_tt(gen, root)
        finally:
            if gen is not None:
                gen.close()
            ttnn.close_mesh_device(mesh)
        return
    from models.autoports.qwen_qwen3_8_flash_next.tests.full_model_hf import build_hf

    tokenizer = AutoTokenizer.from_pretrained(a.checkpoint, local_files_only=True)
    assert tokenizer.chat_template
    suite = Path(os.environ["TT_MODEL_BRINGUP_ROOT"]) / "runtime/readiness_check/vllm_prompts.txt"
    prompts = [s.strip() for s in suite.read_text().split("\n\n") if s.strip()]
    dest = root / "doc/full_model/qualitative_hf.json"
    report = dict(
        hf_model_id="Qwen/Qwen3.8-Flash-Next",
        revision=Path(a.checkpoint).name,
        checkpoint=a.checkpoint,
        tokenizer_class=type(tokenizer).__name__,
        chat_template_present=True,
        prompt_mode="chat",
        rendering="apply_chat_template(tokenize=False,add_generation_prompt=True)",
        source=str(suite),
        generation=dict(
            max_new_tokens=128,
            do_sample=False,
            num_beams=1,
            eos_token_id=sorted(GenerationConfig.from_pretrained(a.checkpoint, local_files_only=True).eos_token_id),
        ),
        prompts=[],
    )
    if dest.exists():
        previous = json.loads(dest.read_text())
        assert previous["revision"] == report["revision"] and previous["generation"] == report["generation"]
        report = previous
    model = build_hf(checkpoint=a.checkpoint)
    for i, prompt in enumerate(prompts):
        if any(row["id"] == i for row in report["prompts"]):
            continue
        rendered = tokenizer.apply_chat_template(
            [dict(role="user", content=prompt)], tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        print("QUALITATIVE_HF_START", i, len(ids), flush=True)
        tick = time.perf_counter()
        with torch.inference_mode():
            tokens = model.generate(
                torch.tensor([ids]),
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = tokens[0, len(ids) :].tolist()
        text = tokenizer.decode(generated, skip_special_tokens=False)
        report["prompts"].append(
            dict(
                id=i,
                prompt=prompt,
                rendered_prompt=rendered,
                prompt_token_ids=ids,
                token_ids=generated,
                completion=text,
                seconds=time.perf_counter() - tick,
            )
        )
        dest.write_text(json.dumps(report, indent=2) + "\n")
        print("QUALITATIVE_HF", i, repr(text), flush=True)


if __name__ == "__main__":
    main()
