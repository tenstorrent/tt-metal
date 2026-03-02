import torch

torch.set_default_device("cpu")

# -------- TRACE FUNCTION --------
TRACE_FILE = "models/experimental/parakeet/tests/parakeet_trace.txt"

# def trace_calls(frame, event, arg):

#     if event != "call":
#         return

#     filename = frame.f_code.co_filename.replace("\\", "/")

#     if "nemo/collections/asr" in filename:

#         func_name = frame.f_code.co_name
#         line_no = frame.f_lineno
#         short_file = os.path.basename(filename)

#         with open(TRACE_FILE, "a") as f:
#             f.write(f"{short_file}:{line_no} -> {func_name}\n")

#     return trace_calls


# -------- YOUR ORIGINAL SCRIPT --------

import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2", map_location="cpu")

asr_model.eval()
output = asr_model.transcribe(["models/experimental/parakeet/tests/2086-149220-0033.wav"])

print(output[0].text)
