# Transformer-based Optical Character Recognition (TrOCR) model

## Introduction

The TrOCR model was proposed in "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models".

The TrOCR model is an encoder-decoder model, consisting of an image Transformer as encoder, and a text Transformer as decoder. The image encoder was initialized from the weights of BEiT, while the text decoder was initialized from the weights of RoBERTa. The VisionEncoderDecoderModel can be used to initialize an image-to-text model with any pretrained Transformer-based vision model as the encoder (e.g. ViT, BEiT, DeiT, Swin) and any pretrained language model as the decoder (e.g. RoBERTa, GPT2, BERT, DistilBERT).


https://huggingface.co/docs/transformers/en/model_doc/trocr


## Details

The entry point of the Fucntional TrOCR model is the trocr fucntion located in ttnn_trocr_causal_lm. The "microsoft/trocr-base-handwritten" version from Hugging Face is utilized as the reference model.

NOTE: For the model, we have used two torch operations: `torch.expand` and `torch.masked_fill`. These operations are used inside the model for generating the attention mask, which currently does not have support in TTNN.


## How to Run

To run the demo for Optical Character Recognition (OCR) with using the TrOCR model, follow these instructions:

- Use the following command to run the ttnn_trocr_causal_lm:
```
pytest --disable-warnings --input-path="models/sample_data/iam_ocr_image.jpg" models/experimental/functional_trocr/demo/ttnn_trocr_demo.py::test_trocr_demo
```

If you wish to run the demo with different input sample , replace <address_to_your_input> with the path for your input in the following command:

```
pytest --disable-warnings --input-path=<address_to_your_input> models/experimental/functional_trocr/demo/ttnn_trocr_demo.py::test_trocr_demo
```

Our second demo is designed to run TrOCR for Optical Character Recognition (OCR) using `Teklia/IAM-line` validation Dataset, run this with the following command:
```
pytest --disable-warnings models/experimental/functional_trocr/demo/ttnn_trocr_demo.py::test_trocr_demo_iam_dataset
```

## Dataset
The IAM Handwriting Database `Teklia/IAM-line` contains forms of handwritten English text which can be used to train and test handwritten text recognizers and to perform writer identification and verification experiments.

## Results
The model inference accuracy for 10 data is 92% on Grayskull and Wormhole devices.
