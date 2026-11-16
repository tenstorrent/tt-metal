from pathlib import Path

p = Path("/root/.cache/uv/environments-v2/parser-466021493977ef71")
python = "/root/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12"
old = "/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/swe-verifier-recovery-env"
for file in (p / "bin").iterdir():
    if file.is_symlink() and file.name.startswith("python"):
        file.unlink()
        file.symlink_to(python if file.name == "python" else "python")
    elif file.is_file():
        b = file.read_bytes()
        if b.startswith(b"#!"):
            file.write_bytes(b.replace(old.encode(), str(p).encode()))
(p / "pyvenv.cfg").write_text(
    "home = "
    + str(Path(python).parent)
    + "\nimplementation = CPython\nversion_info = 3.12.12\ninclude-system-site-packages = false\n"
)
