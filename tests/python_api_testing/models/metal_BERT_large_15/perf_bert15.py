from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from datasets import load_dataset
from loguru import logger
from torchvision import models
from transformers import AutoImageProcessor
from transformers import BertForQuestionAnswering, BertTokenizer, pipeline
from tests.python_api_testing.models.conftest import model_location_generator_

import tt_lib as tt_lib
from utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from test_bert_batch_dram import TtBertBatchDram

from utility_functions import (
    enable_compile_cache,
    enable_compilation_reports,
    enable_memory_reports,
    comp_allclose_and_pcc,
    comp_pcc,
    comp_allclose,
    disable_compile_cache,
    profiler
)

from loguru import logger

BATCH_SIZE = 9
model_name = "phiyodr/bert-large-finetuned-squad2"
tokenizer_name  = "phiyodr/bert-large-finetuned-squad2"
comments = "Large"
seq_len = 384
real_input = True
attention_mask = True
token_type_ids = True
dram = True
model_location_generator = model_location_generator_


def test_perf():
    disable_compile_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device, tt_lib.device.MemoryAllocator.BASIC if dram else tt_lib.device.MemoryAllocator.L1_BANKING)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    mem_config = tt_lib.tensor.MemoryConfig(True, -1, tt_lib.tensor.BufferType.DRAM if dram else tt_lib.tensor.BufferType.L1)

    HF_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    HF_model.eval()
    tt_model = TtBertBatchDram(HF_model.config, HF_model, device, mem_config)

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    context = BATCH_SIZE * [
        "Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. The prophet and founding hero of modern archaeology, Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art."
    ]
    question = BATCH_SIZE * ["What discipline did Winkelmann create?"]
    inputs = tokenizer.batch_encode_plus(
        zip(question, context),
        max_length=seq_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=attention_mask,
        return_token_type_ids=token_type_ids,
        return_tensors="pt"
    )
    tt_input = tt_model.model_preprocessing(**inputs)

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_out = HF_model(**inputs)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model(1, *tt_input)
        tt_lib.device.Synchronize()
        profiler.end(first_key, force_enable=True)
        del tt_output
        tt_input = tt_model.model_preprocessing(**inputs)

        enable_compile_cache()

        profiler.start(second_key)
        tt_output = tt_model(1, *tt_input)
        tt_lib.device.Synchronize()
        profiler.end(second_key, force_enable=True)


    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)

    prep_report("bert15", BATCH_SIZE, first_iter_time, second_iter_time, comments, cpu_time)



def prep_report(model_name: str, batch_size: int, inference_and_compile_time: float, inference_time: float, comments: str, inference_time_cpu: float=None):

    def write_dict_to_file(csv_path, dict_res):
        columns = ", ".join([str(d) for d in dict_res.keys()])
        # values = ", ".join([("{:.2f}".format(d) if isinstance(d, float) else str(d)) for d in dict_res.values()])
        values = ", ".join([d for d in dict_res.values()])

        with open(csv_path, "w") as csvfile:
            csvfile.write(columns)
            csvfile.write("\n")
            csvfile.write(values)


    compile_time = inference_and_compile_time - inference_time
    gs_throughput = batch_size * (1/inference_time)
    cpu_throughput = batch_size * (1/inference_time_cpu) if inference_time_cpu else "unknown"
    cpu_throughput = "{:.5f}".format(cpu_throughput) if not isinstance(cpu_throughput, str) else cpu_throughput
    dict_res = {
        "Model": model_name,
        "Setting": comments,
        "Batch": str(batch_size),
        "First Run (sec)": "{:.2f}".format(inference_and_compile_time),
        "Second Run (sec)":  "{:.2f}".format(inference_time),
        "Compile Time (sec)": "{:.2f}".format(compile_time),
        "Inference Time GS (sec)": "{:.2f}".format(inference_time),
        "Throughput GS (batch*inf/sec)": "{:.2f}".format(gs_throughput),
        "Inference Time CPU (sec)": "{:.2f}".format(inference_time_cpu),
        "Throughput CPU (batch*inf/sec)": cpu_throughput,
    }

    csv_file = f"perf_{model_name}.csv"
    write_dict_to_file(csv_file, dict_res)
