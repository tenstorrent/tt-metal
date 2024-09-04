# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
from loguru import logger
from models.experimental.grok.tt.model_config import TtModelArgs
from models.experimental.grok.reference.model import Transformer
from models.experimental.grok.reference.tokenizer import Tokenizer


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


def main():
    n_layers = 32
    iterations = 20

    # Can avoid running reference model to speed up the test (unless measuring PCC)
    run_ref_pt = True

    model_args = TtModelArgs()
    model_args.n_layers = n_layers

    state_dict = {}
    for i in range(1 + (n_layers - 1) // 4):
        state_dict_i = torch.load(model_args.consolidated_weights_path(i), map_location="cpu")
        state_dict.update(state_dict_i)

    tokenizer = Tokenizer(model_args.tokenizer_path)

    # TODO Update the prompt
    # prompts = ["It_was_the_best_of_times_"] * 32
    prompts = [""] * 32
    # Space token -> (U+2581) == "▁"

    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    if run_ref_pt:
        reference_model = Transformer(args=model_args)
        reference_model.load_state_dict(torch.load(model_args.state_dict_path))

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    generation_start_pos = 0
    generation_length = iterations
    if run_ref_pt:
        all_tests_pass = True

    seqlen = 1  # Generating one token per user at a time
    batch = 32

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)

    # Keep track of generated outputs to print out later
    if run_ref_pt:
        all_outputs_ref = []
        all_logits = []

    # After loading the model weights, wait for an input to start the generation
    # print("Waiting for an input to start...")
    # input()

    for i in range(generation_length):
        print(f"[Decode] Generating token {i}")

        start_pos = generation_start_pos + i

        if run_ref_pt:  # Run reference model
            positions = torch.LongTensor([start_pos])
            print(f"pt_decode_input = {pt_decode_input.shape}")
            ref_output = reference_model(pt_decode_input, positions)  # mask)
            all_logits.append(ref_output)

        print(f"encoded_prompts[0] = {len(encoded_prompts[0])}")

        if run_ref_pt:
            pt_out_tok = torch.argmax(ref_output, dim=-1).squeeze(1)
            pt_decode_input = embd(pt_out_tok).view(batch, seqlen, -1)
            all_outputs_ref.append(pt_out_tok[0].tolist())
            torch.save(all_logits, "ref_logits.pt")

        # TODO Space decoding is currently not working as expected
        # TODO print All 32 users
        if run_ref_pt:
            print("[User 0] Ref generation: ", "".join(tokenizer.decode(all_outputs_ref)))


if __name__ == "__main__":
    main()
