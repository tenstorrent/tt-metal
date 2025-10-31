import argparse
import os
import torch
from models.common.utility_functions import comp_pcc

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir_1")
parser.add_argument("--input_dir_2")
parser.add_argument("--filter_key", default=None)
args = parser.parse_args()

files_1 = os.listdir(args.input_dir_1)
files_2 = os.listdir(args.input_dir_2)
files = list(set(files_1 + files_2))
files = sorted(files, key=lambda x: x.lower())

for filename in files:
    if not filename.startswith("op_") or not filename.endswith(".pt"):
        continue
    if args.filter_key is not None and args.filter_key not in filename:
        continue
    if filename not in files_1 or filename not in files_2:
        print(f"Warning no matching files names {filename}")
        continue

    tensor_1 = torch.load(os.path.join(args.input_dir_1, filename))
    tensor_2 = torch.load(os.path.join(args.input_dir_2, filename))

    if tensor_1.shape != tensor_2.shape:
        print(f"Warning tensor shapes do not match {tensor_1.shape} != {tensor_2.shape} for filename={filename}")
        continue

    if tensor_1.numel() == 0:
        print(f"Skipping zero-sized tensor for filename={filename}")
        continue

    if torch.any(torch.isnan(tensor_1)).item() or torch.any(torch.isnan(tensor_2)).item():
        print(f"Warning tensors contain NaN values for filename={filename}")
        continue

    passed, pcc = comp_pcc(tensor_1, tensor_2, 1.0)
    print(f"filename: {filename}")
    print(
        f"   mean(tensor_1)={torch.mean(tensor_1)}, std(tensor_1)={torch.std(tensor_1)}, max(abs(diff))={torch.max(torch.abs(tensor_1-tensor_2))} pcc={pcc}"
    )
