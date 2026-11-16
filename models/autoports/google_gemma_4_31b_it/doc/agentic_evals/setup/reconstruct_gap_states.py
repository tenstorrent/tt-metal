import ast
import hashlib
import json
import pathlib
import shlex
import subprocess

A = json.load(
    open(
        "models/autoports/google_gemma_4_31b_it/doc/agentic_evals/anomalies/swe85-below-band/correct-actions/remaining-source-audit.private.json"
    )
)


def reconstruct(task, base, pre):
    edits = []
    proof = []
    if task == "psf__requests-1142":
        src = base / "build/lib/requests/models.py"
        dst = pre / "requests/models.py"
        assert src.stat().st_size == 20789
        s = (
            src.read_text()
            .replace("import collections", "import collections\nimport collections.abc")
            .replace("collections.Callable", "collections.abc.Callable")
        )
        dst.write_text(s)
        edits = [
            {
                "files": ["requests/models.py"],
                "native_messages": [3, 51, 52, 53, 55, 56, 61],
                "effect": "Native initial ls lists build/lib; native51 size20789 matches immutable-image build/lib. Native52 copies it into requests/models.py; native56 re-adds compatibility edits. No earlier command modifies build/lib. Image build file hash equals base tracked models.py, independently checked.",
            }
        ]
        proof = [
            {
                "build_file_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                "size_matches_native": True,
                "build_equals_extracted_tracked": src.read_bytes() == (base / "requests/models.py").read_bytes(),
                "native_hash_available": False,
            }
        ]
    else:
        msgs = json.loads(pathlib.Path(A[task]["trajectory"]).read_text())["messages"]
        p = pre / "seaborn/_core/plot.py"

        def command(i):
            return json.loads(msgs[i]["tool_calls"][0]["function"]["arguments"])["command"]

        def sedline(line):
            parts = shlex.split(line)
            assert parts[:2] == ["sed", "-i"] and parts[-1] == "seaborn/_core/plot.py" and len(parts) == 4
            # Only exact audited expression, explicit local path; no shell or agent execution.
            subprocess.run(["sed", "-i", parts[2], str(p)], check=True)

        sedline(command(64).splitlines()[0])
        lines = p.read_text().splitlines(True)
        del lines[1477]
        for i in range(1478, 1485):
            lines[i] = lines[i].replace("                    ", "                ")
        p.write_text("".join(lines))
        # 74 failed shell and its Python fragment contains no write; state unchanged.
        for line in command(76).splitlines():
            if line.startswith("sed -i "):
                sedline(line)
        lines = p.read_text().splitlines(True)
        lines = [
            "                if keep_na:\n"
            if l.startswith("                    if keep_na:")
            else "                " + l[20:]
            if l.startswith(("                    #", "                    present =", "                    nulled ="))
            else l
            for l in lines
        ]
        lines = [
            "                    " + l[16:]
            if l.startswith(
                (
                    "                # The simpler",
                    "                # But that",
                    "                # is written",
                    "                # Matplotlib",
                    "                # Downstream",
                    "                present =",
                    "                nulled =",
                )
            )
            else l
            for l in lines
        ]
        script = command(92).split("cat <<'EOF' > fix_indent.py\n", 1)[1].split("\nEOF", 1)[0]
        # Extract literal lines[index]=value assignments only; do not evaluate or execute agent Python.
        for node in ast.parse(script).body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == "lines"
            ):
                lines[ast.literal_eval(node.targets[0].slice)] = ast.literal_eval(node.value)
        p.write_text("".join(lines))
        for msg, a, b in [(95, 1470, 1490), (99, 1628, 1646)]:
            output = json.loads(msgs[msg]["content"])["output"]
            expected = "".join(f"{i:6}\t{lines[i-1]}" for i in range(a, b + 1))
            proof.append({"native_message": msg, "range": [a, b], "numbered_output_exact_match": output == expected})
            assert output == expected, (msg, output[:100], expected[:100])
        edits = [
            {
                "files": ["seaborn/_core/plot.py"],
                "native_messages": [64, 68, 74, 76, 80, 86, 92],
                "effect": "64 successful first sed despite second error;68 deletion plus literal whitespace rewrite;74 no source write;76 audited sed expressions;80/86 literal prefix transforms;92 literal indexed replacements. Native95 and99 line slices match exactly. No arbitrary agent script executed.",
            }
        ]
    return edits, proof
