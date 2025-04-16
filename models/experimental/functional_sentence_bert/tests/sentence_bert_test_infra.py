# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger
import transformers
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import is_wormhole_b0
from models.experimental.functional_sentence_bert.reference.sentence_bert import BertModel, custom_extended_mask
from models.experimental.functional_sentence_bert.ttnn.ttnn_sentence_bert import ttnn_BertModel, preprocess_inputs
from tests.ttnn.integration_tests.sentence_bert.test_ttnn_sentence_bert import custom_preprocessor
from ttnn.model_preprocessing import preprocess_model_parameters


def load_reference_model(config):
    torch_model = transformers.AutoModel.from_pretrained("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr").eval()
    reference_model = BertModel(config).to(torch.bfloat16)
    reference_model.load_state_dict(torch_model.state_dict())
    return reference_model


def load_ttnn_model(device, torch_model, config):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    ttnn_model = ttnn_BertModel(parameters=parameters, config=config)
    return ttnn_model


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


input = [
    "Yarın tatil yapacağım, ailemle beraber doğada vakit geçireceğiz, yürüyüşler yapıp, keşifler yapacağız, çok keyifli bir tatil olacak.",
    "Yarın tatilde olacağım, ailemle birlikte şehir dışına çıkacağız, doğal güzellikleri keşfedecek ve eğlenceli zaman geçireceğiz.",
    "Yarın tatil planım var, ailemle doğa yürüyüşlerine çıkıp, yeni yerler keşfedeceğiz, harika bir tatil olacak.",
    "Yarın tatil için yola çıkacağız, ailemle birlikte sakin bir yerlerde vakit geçirip, doğa aktiviteleri yapacağız.",
    "Yarın tatilde olacağım, ailemle birlikte doğal alanlarda gezi yapıp, yeni yerler keşfedeceğiz, eğlenceli bir tatil geçireceğiz.",
    "Yarın tatilde olacağım, ailemle birlikte şehir dışında birkaç gün geçirip, doğa ile iç içe olacağız.",
    "Yarın tatil için yola çıkıyoruz, ailemle birlikte doğada keşif yapıp, eğlenceli etkinliklere katılacağız.",
    "Yarın tatilde olacağım, ailemle doğada yürüyüş yapıp, yeni yerler keşfederek harika bir zaman geçireceğiz.",
]


class SentenceBERTTestInfra:
    def __init__(
        self,
        device,
        batch_size,
        model_location_generator=None,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        config = transformers.BertConfig.from_pretrained("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
        tokenizer = transformers.AutoTokenizer.from_pretrained("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
        encoded_input = tokenizer(input, padding="max_length", max_length=384, truncation=True, return_tensors="pt")
        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]
        print("att is", attention_mask.shape)
        extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
        print("ext is", extended_mask.shape)
        token_type_ids = encoded_input["token_type_ids"]
        position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int64).unsqueeze(dim=0)

        self.torch_input_tensor = encoded_input["input_ids"]
        reference_model = load_reference_model(config)
        (
            self.ttnn_input_ids,
            self.ttnn_token_type_ids,
            self.ttnn_position_ids,
            self.ttnn_attention_mask,
        ) = preprocess_inputs(input_ids, token_type_ids, position_ids, extended_mask, device)
        p(self.ttnn_input_ids, "input")
        p(self.ttnn_attention_mask, "ttnn_attention_mask")
        p(self.ttnn_token_type_ids, "ttnn_token_type_ids")
        p(self.ttnn_position_ids, "ttnn_position_ids")
        # ss
        self.ttnn_model = load_ttnn_model(self.device, reference_model, config)
        print("inputsare", input_ids.shape, extended_mask.shape, token_type_ids.shape, position_ids.shape)
        print("ttnn inputsare", input_ids.shape, extended_mask.shape, token_type_ids.shape, position_ids.shape)
        torch_out = reference_model(
            input_ids, attention_mask=extended_mask, token_type_ids=token_type_ids, position_ids=position_ids
        )
        self.torch_output_tensor_1, self.torch_output_tensor_2 = torch_out.last_hidden_state, torch_out.pooler_output

    def run(self):
        p(self.input_tensor, "input tensor is")
        self.output_tensor_1, self.output_tensor_2 = self.ttnn_model(
            self.input_tensor,
            self.ttnn_attention_mask,
            self.ttnn_token_type_ids,
            self.ttnn_position_ids,
            device=self.device,
        )

    def setup_l1_sharded_input(self, device, torch_input_tensor=None):
        if is_wormhole_b0():
            core_grid = ttnn.CoreGrid(y=8, x=8)
        else:
            exit("Unsupported device")
        # torch tensor
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        # n, c, h, w = torch_input_tensor.shape
        # # sharded mem config for fold input
        # num_cores = core_grid.x * core_grid.y
        # shard_h = (n * w * h + num_cores - 1) // num_cores
        # grid_size = core_grid
        # grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
        # shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        # shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, 16), ttnn.ShardOrientation.ROW_MAJOR)
        # input_mem_config = ttnn.MemoryConfig(
        #     ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        # )
        # torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
        # torch_input_tensor = torch_input_tensor.reshape(1, 1, h * w * n, c)
        # tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        # tt_inputs_host = ttnn.pad(tt_inputs_host, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)
        torch_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.uint32)
        input_memory_config = ttnn.create_sharded_memory_config(
            torch_input_tensor.shape,
            core_grid=device.core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
        )

        return torch_input_tensor, input_memory_config

    def setup_dram_sharded_input(self, device, torch_input_tensor=None, mesh_mapper=None, mesh_composer=None):
        tt_inputs_host, input_mem_config = self.setup_l1_sharded_input(device)
        dram_grid_size = device.dram_grid_size()
        # dram_shard_spec = ttnn.ShardSpec(
        #     ttnn.CoreRangeSet(
        #         {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        #     ),
        #     [
        #         divup(tt_inputs_host.volume() // tt_inputs_host.shape[-1], (dram_grid_size.x * dram_grid_size.y)),
        #         16,
        #     ],
        #     ttnn.ShardOrientation.ROW_MAJOR,
        # )
        # sharded_mem_config_DRAM = ttnn.MemoryConfig(
        #     ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        # )
        sharded_mem_config_DRAM = ttnn.create_sharded_memory_config(
            tt_inputs_host.shape,
            core_grid=device.core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
        )
        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def validate(self, output_tensor=None):
        output_tensor_1 = ttnn.to_torch(self.output_tensor_1)
        valid_pcc = 0.94
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor_1, output_tensor_1, pcc=valid_pcc)

        logger.info(f"sentence bert batch_size={self.batch_size}, PCC={self.pcc_message}")

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensor_1)
        # ttnn.deallocate(self.output_tensor_2)


def create_test_infra(
    device,
    batch_size,
):
    return SentenceBERTTestInfra(
        device,
        batch_size,
    )
