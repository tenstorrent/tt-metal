import glob
import json
import subprocess
import sys

vids = sorted(glob.glob(sys.argv[1] + "/**/*.mp4", recursive=True))
bad = []
for v in vids:
    code = "import sys;from vllm.multimodal.video import Molmo2VideoBackend as B;f,m=B.load_bytes(open(sys.argv[1],'rb').read(),num_frames=384,frame_sample_mode='uniform_last_frame',max_fps=2);print(f.shape)"
    try:
        r = subprocess.run(
            ["/home/ubuntu/ssinghal/molmo2/serve_env/bin/python", "-c", code, v], capture_output=True, timeout=900
        )
        if r.returncode != 0:
            bad.append((v, r.returncode))
    except subprocess.TimeoutExpired:
        bad.append((v, "timeout"))
print(len(vids), "videos; bad:", bad)
json.dump(bad, open(sys.argv[2], "w"))
