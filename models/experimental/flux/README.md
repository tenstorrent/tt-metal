# Flux.1

This is an implementation of Flux.1 \[schnell\], which is a variant of [Flux.1](https://blackforestlabs.ai/announcing-black-forest-labs/), a generative model for image synthesis guided by text prompts.

The model consists of two different text encoders together with their tokenizers, a scheduler, a trasformer and a VAE.

## Running the Tests

The tests are executed using the following command:

```sh
pytest models/experimental/flux/tests
```

## Running the Demo

The demo is executed using the following command:

```sh
pytest models/experimental/flux/demo.py
```
