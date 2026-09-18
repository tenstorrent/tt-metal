from pathlib import Path

from support import require


def process_identity(pid, proc=Path("/proc")):
    text = (Path(proc) / str(pid) / "stat").read_text()
    end = text.rfind(")")
    fields = text[end + 2 :].split()
    require(end >= 0 and len(fields) > 19, "Malformed process identity")
    return dict(pid=int(pid), start_ticks=int(fields[19]), state=fields[0])
