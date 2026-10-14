import hashlib, http.server, json, os, sys, tempfile, threading, unittest
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "source"))
from stage_checkpoint import SOURCE_HASHES, load_runtime

assert sys.version_info[:2] == (3, 12), sys.version
RuntimeDownload, runtime_verify, runtime_origins = load_runtime(Path.cwd() / "source")
from models.demos.blackhole.qwen38_flash_next.tools import download_checkpoint as download
from models.demos.blackhole.qwen38_flash_next.tools import verify_release as verify

origins = {}
for name, digest in SOURCE_HASHES.items():
    p = Path.cwd() / "source/models/demos/blackhole/qwen38_flash_next" / name
    assert hashlib.sha256(p.read_bytes()).hexdigest() == digest
    origins[name] = {"path": str(p), "sha256": digest}
assert (
    Path(download.__file__).resolve()
    == (Path.cwd() / "source/models/demos/blackhole/qwen38_flash_next/tools/download_checkpoint.py").resolve()
)
assert not any(n == "ttnn" or n.startswith("ttnn.") for n in sys.modules)
payload = b"original downloader loopback compatibility check\n"


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        assert self.headers["Range"] == f"bytes=0-{len(payload)-1}"
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    out = root / "checkpoint"
    out.mkdir()
    work = root / "work"
    work.mkdir()
    row = {"Path": "fixture.bin", "Size": len(payload), "Sha256": hashlib.sha256(payload).hexdigest()}
    download.FILE_URL = f"http://127.0.0.1:{server.server_port}/fixture.bin"
    obj = download.Download(download.RELEASE_REVISION, [row], out, work, 1)
    assert obj.run() == 0
    assert (out / "fixture.bin").read_bytes() == payload
    assert verify._verify_file(out, row)["sha256"] == row["Sha256"]
    (out / "fixture.bin").write_bytes(payload[:-1] + b"!")
    try:
        verify._verify_file(out, row)
    except ValueError:
        pass
    else:
        raise AssertionError("original verifier accepted corruption")
server.shutdown()
server.server_close()
fds = {}
for p in Path("/proc/self/fd").iterdir():
    try:
        t = os.readlink(p)
    except FileNotFoundError:
        continue
    if t.startswith("/dev/tenstorrent/"):
        fds[p.name] = t
assert not fds
Path("evidence").mkdir(exist_ok=True)
Path("evidence/runtime.json").write_text(
    json.dumps(
        {
            "python": sys.version,
            "origins": origins,
            "actual_stage_runtime_loader": runtime_origins,
            "ttnn_imported": False,
            "fds": fds,
            "original_download_loopback": "PASS",
            "original_verify_corruption_refusal": "PASS",
        },
        indent=2,
    )
)
suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_stage.py")
r = unittest.TextTestRunner(verbosity=2).run(suite)
assert r.wasSuccessful()
