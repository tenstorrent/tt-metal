# t5 Model

## Introduction
Text-To-Text Transfer Transformer (T5)  is an encoder-decoder model which reframes all NLP tasks into a unified text-to-text-format where the input and output are always text strings.
With this model, tasks are seamlessly executed by adding a specific prefix to the input. For instance, summarization tasks begin with 'summarize:'. This flexibility enables T5 to excel across a wide range of tasks.

## Details

The entry point to the T5 model is `t5_for_conditional_generation` in `ttnn_optimized_functional_t5.py`. The model picks up certain configs and weights from huggingface pretrained model. `t5-small` and `google/flan-t5-small` versions from huggingface are used as reference.


# t5 Model for Conditional Generation
In the demo of conditional generation, the model accepts input text, and it provides a summarized version of the input text.

## Inputs
Inputs for the demo are provided from `input_data.json` by default. If you need to change the inputs or provide a different path, modify the input_path parameter in the command accordingly. We recommend against modifying the input_data.json file directly.

## How to Run

- Use the following command to run T5 for conditional generation demo using `t5-small`.
```
pytest --disable-warnings --input-path="models/demos/grayskull/t5/demo/input_data_cg.json" models/demos/grayskull/t5/demo/conditional_generation_demo.py::test_t5_demo_for_summarize[8-128-64-t5-small]
```

- Alternatively, use the following command to run T5 for conditional generation demo using `google/flan-t5-small`.
```
pytest --disable-warnings --input-path="models/demos/grayskull/t5/demo/input_data_cg.json" models/demos/grayskull/t5/demo/conditional_generation_demo.py::test_t5_demo_for_summarize[8-128-64-google/flan-t5-small]
```

- If you wish to run the demo with a different input file, replace <address_to_your_json_file.json> with the path to your JSON file in the following command:
```
pytest --disable-warnings --input-path=<address_to_your_json_file.json> models/demos/grayskull/t5/demo/conditional_generation_demo.py::test_t5_demo_for_summarize[8-128-64-t5-small]
```

Second demo is designed to run with `openai/summarize_from_feedback` dataset. The dataset includes human feedback which is used as input text to the model and summary of the feedback is used to validate the model.

- Use the following command to run the second demo of T5 using `t5-small` variant for summarize the input text demo.
```
pytest --disable-warnings models/demos/grayskull/t5/demo/conditional_generation_demo.py::test_t5_demo_for_summarize_dataset[8-128-64-t5-small]
```

- Alternatively, use the following command to run the second demo of T5 using `google/flan-t5-small` variant for summarize the input text demo.
```
pytest --disable-warnings models/demos/grayskull/t5/demo/conditional_generation_demo.py::test_t5_demo_for_summarize_dataset[8-128-64-google/flan-t5-small]
```

## Results
The input is fed into the T5 model for conditional generation, and the output will be a summarized and simplified version of the input text.


# t5 model for Question Answering
In the demo of Question Answering, the model accepts input text as context and a question. Subsequently, the model generates an answer corresponding to the question within the provided context. The input is formatted with the prefixes "context" and "question" to distinguish between the context and the question respectively.

## Inputs
Inputs by default are provided from `input_data_qa.json`. If you wish you to change the inputs or provide a different path to `test_functional_t5_demo`.

## How to run

- Use the following command to run the functional t5 demo for Question and Answering.
```
pytest --disable-warnings --input-path="models/demos/grayskull/t5/demo/input_data_qa.json" models/demos/grayskull/t5/demo/question_answering_demo.py::test_functional_t5_demo
```

- If you wish to run the demo with a different input use
```
pytest --disable-warnings --input-path="[address_to_your_json_file]"  models/demos/grayskull/t5/demo/question_answering_demo.py::test_functional_t5_demo
```
This file is expected to have exactly 8 inputs.

- Our second demo is designed to run SQuADV2 dataset, run this with
```
pytest --disable-warnings models/demos/grayskull/t5/demo/question_answering_demo.py::test_functional_t5_demo_squadv2
```

## Results

The T5 model receives input in the form of a question along with its context. The T5 model then generates an answer corresponding to the question posed within that context.
