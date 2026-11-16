import hashlib
import json
import pathlib
import subprocess

from transformers import AutoTokenizer

D = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
F = D / "anomalies/swe85-below-band/full-replies-update2"
O = D / "anomalies/swe85-pytest5787-controls"
O.mkdir(exist_ok=True)
O.chmod(0o700)
rev = "842da3794eaa0b77d5f08bae87a17459d91ff475"
tok = AutoTokenizer.from_pretrained(
    "/home/mvasiljevic/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/" + rev,
    revision=rev,
    local_files_only=True,
)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, j, private=True):
    p.write_text(json.dumps(j, indent=2) + "\n")
    p.chmod(0o600 if private else 0o644)


jobs = []
man = []
for n in [1, 65]:
    name = f"pytest-dev__pytest-5787.call{n}"
    pkpath = F / (name + ".packet.private.json")
    casepath = F / (name + ".case.private.json")
    pk = json.loads(pkpath.read_text())
    full = json.loads(casepath.read_text())
    text = pk["reconstructed_full_reply"]
    native = pk["native_response"]
    msg = native["choices"][0]["message"]
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    if n == 1:
        marker = "GetAll-bash"
        pos = text.index(marker)
        why = "Visible content emits GetAll-bash instead of structured tool call and simulates repeated bash/tool exchanges. First response omitted from native assistant transcript; server response recovered via request/task mapping. Canonical reasoning field contains simulated thought exchanges too; parser-flattened ordering cannot locate the first native generated anomaly exactly."
    else:
        marker = "I will’ll"
        pos = text.index(marker) + len("I will")
        why = "At native message129 tool command comment, duplicated curly apostrophe/contraction in I will’ll; nearby a a string duplication lies in same short window. Localized text anomaly per root, not whole-reply collapse or proven cause of invalid source edit."
    idx = next(i for i, (a, b) in enumerate(enc["offset_mapping"]) if a <= pos < b)
    ids = enc["input_ids"]
    chosen = (ids + [pk["native_stop_id"]])[idx : idx + 64]
    assert len(chosen) == 64
    request = json.loads(pk["raw_request_body"])
    sampling = {k: request.get(k) for k in ["temperature", "top_p", "top_k"]}
    sampling = {k: v for k, v in sampling.items() if v is not None}
    metadata = {
        **full["metadata"],
        "scope": "Full reconstructed native request plus complete canonical reply prefix to onset; chosen64 positions. Supplementary anomaly onset test, not full-reply agreement or completed reproduction.",
        "onset_char_offset": pos,
        "onset_token_index": idx,
        "motif": marker,
        "onset_evidence": why,
        "seed": 42,
        "native_seed": request.get("seed"),
        "sampling": sampling,
        "seed_caveat": "Native seed absent unless recorded; diagnostic42/43/44 are new trials, not exact native replay.",
        "source_full_reply_case": str(casepath),
        "raw_packet": str(pkpath),
        "raw_packet_sha256": sha(pkpath),
        "correct_continuation_status": "not constructed; no invented model answer",
        "executed": False,
    }
    p = O / (name + ".onset.private.json")
    write(
        p,
        {
            "prompt_token_ids": full["prompt_token_ids"] + ids[:idx],
            "continuations": {"chosen": chosen},
            "metadata": metadata,
        },
    )
    jobs.append(
        {"name": name + ".onset", "case": str(p), "branches": ["chosen"], "free_run_tokens": 256, "free_run_trials": 3}
    )
    man.append(
        {
            "call": n,
            "response_id": metadata["response_id"],
            "private_case": str(p),
            "private_case_sha256": sha(p),
            "raw_packet": str(pkpath),
            "raw_packet_sha256": sha(pkpath),
            "native_prompt_tokens": metadata["native_prompt_tokens"],
            "reconstructed_request_prompt_tokens": metadata["reconstructed_prompt_tokens"],
            "native_reply_tokens": metadata["native_completion_tokens"],
            "reconstructed_reply_tokens": metadata["reconstructed_completion_tokens"],
            "conditioning_tokens": len(full["prompt_token_ids"]) + idx,
            "onset_positions": len(chosen),
            "onset_token_index": idx,
            "motif": marker,
            "evidence": why,
            "limitations": metadata["reconstruction"],
            "native_message_indices": metadata["native_message_indices"],
            "sampling": sampling,
            "native_seed": request.get("seed"),
            "diagnostic_seeds": [42, 43, 44],
        }
    )
write(O / "cases.json", jobs, False)
commands = {}
validation = []
for mode, config in [("served", "head_bfp8_build_kwargs.json"), ("full", "full_precision_build_kwargs.json")]:
    argv = [
        "python_env/bin/python",
        "-m",
        "models.autoports.google_gemma_4_31b_it.tools.precision_repro_batch",
        "--cases",
        str(O / "cases.json"),
        "--build-kwargs-file",
        str(D / "setup" / config),
        "--output-dir",
        str(O / mode),
    ]
    if mode == "full":
        argv += ["--compare-dir", str(O / "served")]
    r = subprocess.run(argv + ["--validate-only"], capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout, r.stderr)
    commands[mode] = argv
    validation.append(
        {
            "mode": mode,
            "config": str(D / "setup" / config),
            "config_sha256": sha(D / "setup" / config),
            "exit_code": r.returncode,
            "stdout": r.stdout.strip(),
        }
    )
write(
    O / "manifest.json",
    {
        "status": "prepared_CPU_only_not_executed",
        "task": "pytest-dev__pytest-5787",
        "cases": man,
        "cpu_validation": validation,
        "commands_staged_only": commands,
        "no_inference": True,
        "phase_scope": "Anomaly controls independent of fetched-solution score; no score or corrected answer claim.",
    },
    False,
)
# Keep full anomaly packets for provenance, but only fetch decisions in below-band job list.
fulljobs = json.loads((F / "cases.json").read_text())
write(F / "all-packet-cases.archival.json", fulljobs, False)
write(F / "cases.json", [j for j in fulljobs if not j["name"].endswith((".call1", ".call65"))], False)
print(json.dumps({"cases": man, "validation": validation}))
