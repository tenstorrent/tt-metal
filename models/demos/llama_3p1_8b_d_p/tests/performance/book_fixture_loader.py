# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Small stdlib loader for frozen book-token manifests; imports no tensor library."""
import hashlib
import json
import re
from pathlib import Path

from models.demos.llama_3p1_8b_d_p.tests.performance.book_selection import require, validate_source_binding

CONTEXTS = [4096, 8192, 16384, 32768, 65536, 131072]
SCOPE = "raw_book_continuation_next_token_observation"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_book_fixture(manifest_path, manifest_sha256, context_length, checkpoint_path):
    path = Path(manifest_path).resolve()
    require(sha(path) == manifest_sha256, "Book fixture manifest changed")
    manifest = json.loads(path.read_text())
    require(
        manifest.get("schema_version") == 1 and manifest.get("validation_passed") is True,
        "Fixture manifest is not validated",
    )
    require(
        manifest.get("scope") == SCOPE and manifest.get("contexts") == CONTEXTS and manifest.get("slots") == [0, 1],
        "Wrong fixture scope/inventory",
    )
    require(type(context_length) is int and context_length in CONTEXTS, "Unsupported context")
    require(Path(checkpoint_path).resolve() == Path(manifest["checkpoint_path"]).resolve(), "Checkpoint path differs")
    pins = manifest["tokenizer_files_sha256"]
    require(
        set(pins) == {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"},
        "Incomplete checkpoint metadata pins",
    )
    for name, digest in pins.items():
        require(sha(Path(checkpoint_path) / name) == digest, "Checkpoint metadata changed: " + name)
    config = json.loads((Path(checkpoint_path) / "config.json").read_text())
    require(
        config["vocab_size"] == 128256 and config["max_position_embeddings"] == 131072,
        "Checkpoint context/vocabulary differs",
    )
    source_path = Path(manifest["source_contract_path"])
    require(sha(source_path) == manifest["source_contract_sha256"], "Book source contract changed")
    books = json.loads(source_path.read_text())["books"]
    require(len(books) == 2 and [b["slot"] for b in books] == [0, 1], "Wrong source slots")
    for book in books:
        validate_source_binding(book)
    entries = manifest["fixtures"]
    require(
        len(entries) == 12
        and {(e["slot"], e["context_length"]) for e in entries} == {(s, c) for s in (0, 1) for c in CONTEXTS},
        "Missing/duplicate book fixtures",
    )
    wanted = {"tokenizer-validation.json"}
    for entry in entries:
        prefix = f"slot{entry['slot']}-C{entry['context_length']}"
        require(
            entry["token_ids_file"] == prefix + "-token-ids.json"
            and entry["metadata_file"] == prefix + "-metadata.json",
            "Noncanonical fixture filename",
        )
        wanted.update([entry["token_ids_file"], entry["metadata_file"]])
    require(set(manifest["files_sha256"]) == wanted, "Incomplete fixture file inventory")
    for name, digest in manifest["files_sha256"].items():
        require(sha(path.parent / name) == digest, "Fixture file changed: " + name)
    for entry in entries:
        for kind in ("token_ids", "metadata"):
            require(
                manifest["files_sha256"][entry[kind + "_file"]] == entry[kind + "_sha256"],
                "Fixture hash declarations disagree",
            )
    receipt = json.loads((path.parent / "tokenizer-validation.json").read_text())
    require(
        receipt["validation_passed"] is True
        and receipt["status"] == "book_fixture_validation_passed"
        and receipt["failures"] == [],
        "Tokenizer receipt failed",
    )
    require(
        receipt["tokenizer_loaded"] is True
        and receipt["tokenizer_execution_completed"] is True
        and receipt["cpu_tensor_library_imported"] is True,
        "Tokenizer did not complete",
    )
    cpu = receipt["cpu_tensor_library"]
    require(
        cpu["cuda"] is None and (cpu["intra_threads"], cpu["inter_threads"]) == (4, 1), "Wrong CPU tokenizer runtime"
    )
    for key in ("weights_loaded", "model_instantiated", "model_forward_executed", "model_executed", "device_execution"):
        require(receipt[key] is False, "Unexpected execution in tokenizer receipt")
    require(
        receipt["fixtures"] == entries
        and receipt["checkpoint_path"] == manifest["checkpoint_path"]
        and receipt["tokenizer_files_sha256"] == pins,
        "Receipt/manifest mismatch",
    )
    slots = []
    for slot in (0, 1):
        entry = next(e for e in entries if (e["slot"], e["context_length"]) == (slot, context_length))
        ids = json.loads((path.parent / entry["token_ids_file"]).read_text())
        meta = json.loads((path.parent / entry["metadata_file"]).read_text())
        require(
            isinstance(ids, list)
            and len(ids) == context_length
            and all(type(i) is int and 0 <= i < 128256 for i in ids),
            "Invalid exact-C token IDs",
        )
        bos = config["bos_token_id"]
        require(ids[0] == bos and ids.count(bos) == 1, "Expected exactly one initial BOS")
        expected = dict(
            schema_version=1,
            scope=SCOPE,
            slot=slot,
            context_length=context_length,
            prompt_tokens=context_length,
            final_prompt_position=context_length - 1,
            bos_token_id=bos,
            bos_count=1,
            vocab_size=128256,
            token_ids_file=entry["token_ids_file"],
            token_ids_sha256=entry["token_ids_sha256"],
            checkpoint_path=manifest["checkpoint_path"],
            tokenizer_files_sha256=pins,
            output_headroom=0,
            observation_only=True,
        )
        expected.update({k: books[slot][k] for k in ("book_id", "title", "source_sha256", "body_sha256")})
        require(all(meta.get(k) == v for k, v in expected.items()), "Fixture metadata binding differs")
        start, end = meta["book_token_range"]
        require(
            type(start) is int
            and type(end) is int
            and 0 <= start < 512
            and end - start == context_length - 1
            and meta["candidate_offset"] == start,
            "Invalid deterministic book token range",
        )
        token = meta["expected_next_token_id"]
        require(type(token) is int and 0 <= token < 128256 and token != bos, "Invalid expected next token")
        require(
            re.fullmatch(r"\s+[A-Za-z]+", meta["expected_next_token"])
            and meta["expected_next_word"] == meta["expected_next_token"].lstrip(),
            "Invalid whole-word observation",
        )
        require(
            isinstance(meta["trailing_prompt_text"], str)
            and meta["actual_following_text"].startswith(meta["expected_next_token"]),
            "Missing actual book continuation",
        )
        slots.append(dict(token_ids=ids, metadata=meta))
    require(slots[0]["token_ids"] != slots[1]["token_ids"], "Book slots are identical")
    return dict(manifest=manifest, slots=slots)
