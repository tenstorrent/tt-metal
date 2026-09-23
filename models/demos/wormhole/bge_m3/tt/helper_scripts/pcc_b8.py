"""Print the B8/S512 end-to-end PCC.

Mirrors test_model_multibatch._run_full_end_to_end exactly (same seed, same
input construction, same dtype rule) but prints the PCC instead of only
asserting it, so each experiment cell yields a comparable number.
"""

import torch
import transformers

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.bge_m3.tests.test_utils import to_torch, to_ttnn_ids
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

BATCH, SEQ = 8, 512
MODEL_ID = "BAAI/bge-m3"
THRESHOLD = 0.94

device = ttnn.open_device(device_id=0)

hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).eval()
backbone = hf.roberta if hasattr(hf, "roberta") else hf
state_dict = hf.state_dict()

model_args, tt_model, _ = create_tt_model(
    mesh_device=device,
    max_batch_size=BATCH,
    max_seq_len=SEQ,
    dtype=ttnn.bfloat8_b,
    state_dict=state_dict,
    hf_model_name=MODEL_ID,
)

torch.manual_seed(42)
input_ids = torch.randint(low=0, high=model_args.vocab_size, size=(BATCH, SEQ), dtype=torch.long)
non_pad = (int(model_args.pad_token_id) + 1) % model_args.vocab_size
input_ids[input_ids == model_args.pad_token_id] = non_pad
token_type_ids = torch.zeros_like(input_ids)

with torch.no_grad():
    reference = (
        backbone(
            input_ids=input_ids,
            attention_mask=None,
            token_type_ids=token_type_ids,
            position_ids=None,
            return_dict=True,
        )
        .last_hidden_state.unsqueeze(1)
        .to(torch.float32)
    )

tt_out = tt_model.forward(
    input_ids=to_ttnn_ids(input_ids, device),
    attention_mask=None,
    token_type_ids=to_ttnn_ids(token_type_ids, device),
)
tt_torch = to_torch(tt_out, expected_shape=(BATCH, 1, SEQ, model_args.dim))

passing, msg = comp_pcc(reference, tt_torch, THRESHOLD)
print("CELL_PCC %s" % msg)
print("CELL_PASS %s" % ("PASS" if passing else "FAIL"))
ttnn.close_device(device)
