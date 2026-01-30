# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os

import torch
import transformers
from loguru import logger

from models.common.utility_functions import comp_pcc, is_close


def compare_tensors(ref_tensor, tensor, layer_name, pcc, rtol, atol):
    if isinstance(ref_tensor, transformers.cache_utils.DynamicCache):
        logger.info("Skipping DynamicCache check to speedup comparison")
        return True, True

    if (not isinstance(ref_tensor, torch.Tensor)) or (not isinstance(tensor, torch.Tensor)):
        logger.debug(f"Skipping {layer_name}, not torch.Tensor: {type(ref_tensor)} and {type(tensor)}")
        return True, True

    is_passed_pcc, pcc = comp_pcc(ref_tensor, tensor, pcc=pcc)
    is_passed_close = is_close(ref_tensor, tensor, rtol=rtol, atol=atol)
    logger.debug(f"{layer_name} PCC passed: {is_passed_pcc} ({pcc})")
    logger.debug(f"{layer_name} is_close passed: {is_passed_close}")
    return bool(is_passed_pcc), is_passed_close.item()


def compare_inputs_outputs(ref_sample, sample, pcc, rtol, atol, prefix=""):
    """Takes output of def extract_layer_inputs_outputs as an input.
    Returns {
        "args": [is_passed],
        "kwargs": {key: is_passed},
        "output": [is_passed]
    }"""

    ref_layer_name, ref_args, ref_kwargs, ref_output = ref_sample
    layer_name, args, kwargs, output = sample
    assert ref_layer_name == layer_name, f"Different layer names '{ref_layer_name}' and '{layer_name}'"
    assert len(ref_args) == len(args), f"Different args length: {len(ref_args)} and {len(args)}"
    assert set(ref_kwargs.keys()) == set(kwargs.keys()), f"Different kwargs: {ref_kwargs.keys()} and {kwargs.keys()}"
    assert type(ref_output) == type(output)

    ref_layer_name = prefix + ref_layer_name
    layer_name = prefix + layer_name

    compare_results = {
        "args": [],
        "kwargs": {},
        "output": [],
    }

    # args comparison
    for ref_arg, arg in zip(ref_args, args):
        is_passed_pcc, is_passed_close = compare_tensors(ref_arg, arg, f"{layer_name}-->args", pcc, rtol, atol)
        compare_results["args"].append(is_passed_pcc and is_passed_close)

    # kwargs comparison
    for key in ref_kwargs:
        ref_kwarg = ref_kwargs[key]
        kwarg = kwargs[key]
        is_passed_pcc, is_passed_close = compare_tensors(
            ref_kwarg, kwarg, f"{layer_name}-->kwargs-->{key}", pcc, rtol, atol
        )
        compare_results["kwargs"][key] = is_passed_pcc and is_passed_close

    # output comparison
    if isinstance(ref_output, torch.Tensor):
        ref_output = (ref_output,)
        output = (output,)
    elif isinstance(ref_output, transformers.modeling_outputs.BaseModelOutputWithPast):
        ref_output = tuple(ref_output.values())
        output = tuple(output.values())

    assert len(ref_output) == len(output), f"Different outputs '{len(ref_output)}' and '{len(output)}'"
    for ref_out, out in zip(ref_output, output):
        is_passed_pcc, is_passed_close = compare_tensors(ref_out, out, f"{layer_name}-->output", pcc, rtol, atol)
        compare_results["output"].append(is_passed_pcc and is_passed_close)

    return compare_results


def extract_layer_inputs_outputs(x):
    """Structure:

    x.pt structure: [token_samples]
        len(x) == 1 for prefill
        len(x) == num_decode_tokens (32 by default) for decode

    token_sample structure: [expert_samples]
        len(token_sample) == 1
        len(token_sample) == 256 for "*.mlp.*" files in prefill mode
        len(token_sample) == num_experts_per_tok for "*.mlp.*" files in decode mode

    expert_sample structure: (layer_name: str, args: tuple[torch.Tensor], kwargs: dict, output)
    """
    # assert len(x) in (1, 32), len(x)
    for token_sample in x:
        # assert len(token_sample) in (1, 8, 256), len(token_sample)
        token_sample.sort(key=lambda sample: sample[0])
        for expert_sample in token_sample:
            assert len(expert_sample) == 4
            layer_name, args, kwargs, output = expert_sample
            assert isinstance(layer_name, str)
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)
            assert isinstance(output, (torch.Tensor, tuple, transformers.modeling_outputs.BaseModelOutputWithPast))

            yield expert_sample


def compare_files(reference_directory, directory, pcc, rtol, atol):
    reference_files = sorted(os.listdir(reference_directory))
    files = sorted(os.listdir(directory))
    assert reference_files == files, "Files list is different"

    non_passed_fn = []
    for fn in reference_files:
        logger.info(f"Comparing {fn} ...")
        try:
            ref_pt = torch.load(os.path.join(reference_directory, fn), weights_only=False)
            pt = torch.load(os.path.join(directory, fn), weights_only=False)

            passed = True
            for ref_sample, sample in zip(extract_layer_inputs_outputs(ref_pt), extract_layer_inputs_outputs(pt)):
                comparison_results = compare_inputs_outputs(ref_sample, sample, pcc, rtol, atol, prefix=f"{fn}__")

                sample_passed = True
                for res in comparison_results.values():
                    if isinstance(res, dict):
                        res = res.values()
                    sample_passed &= all(res)
                passed &= sample_passed

                logger.debug(f"Results for {fn}__{ref_sample[0]}. Passed: {sample_passed}")
                if not sample_passed:
                    logger.error(f"{fn}__{ref_sample[0]} failed")
                    logger.error(json.dumps(comparison_results, indent=4))
                logger.debug("-" * 40)

        except Exception as e:
            logger.error(f"{fn} failed with {e}")
            passed = False

        if not passed:
            non_passed_fn.append(fn)

        logger.info(f"Results for {fn}. Passed: {passed}")
        logger.info("=" * 80)

    assert len(non_passed_fn) == 0, f"Files which didn't pass the check {non_passed_fn}"
    logger.info("All checks passed")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference_dir",
        type=str,
        required=True,
        help="Original dir with Deepseek inputs_outputs, e.g. /mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache",
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Generate dir with Deepseek inputs_outputs, e.g. /mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache-new",
    )
    parser.add_argument("--pcc", type=float, required=False, default=0.99, help="PCC threshold for comparison check")
    parser.add_argument(
        "--rtol",
        type=float,
        required=False,
        default=1e-2,
        help="rtol threshold for comparison check using torch.isclose-like function. By default is relatively high to do mostly PCC check",
    )
    parser.add_argument(
        "--atol",
        type=float,
        required=False,
        default=1e-2,
        help="atol threshold for comparison check using torch.isclose-like function. By default is relatively high to do mostly PCC check",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    compare_files(args.reference_dir, args.dir, args.pcc, args.rtol, args.atol)


if __name__ == "__main__":
    main()
