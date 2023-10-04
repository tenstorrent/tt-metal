.. _TT-Metal Models BERT Demo:

BERT Demo
=========

Model
-----

This model is based on BERT-Large-uncased and finetuned for qustion answering on `SQuAD2.0 <https://rajpurkar.github.io/SQuAD-explorer/>`_ data set.
The model uses weights from HuggingFace `phiyodr/bert-large-finetuned-squad2 <https://huggingface.co/phiyodr/bert-large-finetuned-squad2>`_.

The model is set up to process a batch of 9 inputs at a time and uses sequence length of 384.
It usees `BFLOAT16` data type and stores some tensors in `L1` memory.


For each input, the model will answer the associated question based only on information provided in the context.
First, both context and quesiton will be tokenized and encoded during pre-processing.
The model will run inference and return a tensor with logits for start and end position of answer in the context.
In post processing, part of context between start and end positions will be extracted from context and returned as a string.

For example, with context
    Johann Joachim Winckelmann was a German art historian and archaeologist. He was a pioneering Hellenist who first articulated the difference between Greek, Greco-Roman and Roman art. The prophet and founding hero of modern archaeology, Winckelmann was one of the founders of scientific archaeology and first applied the categories of style on a large, systematic basis to the history of art.

and question
    What discipline did Winkelmann create?

model would return 274 as start position and 296 as end position, which in post processing will be resolved into answer string
    scientific archaeology


Prerequisites
-------------

Ensure that you have the base TT-Metal source and environment configuration
:ref:`built and ready<Getting Started>`.

Now, from the project root, get the Python virtual environment in which you'll
be working in ready.

::

    source build/python_env/bin/activate

Set ``PYTHONPATH`` to the root for running models. This is a common practice.

::

    export PYTHONPATH=$(pwd)

Running Demo
------------

BERT Demo can by run via ``pytest`` by running the following

::

    pytest models/experimental/metal_BERT_large_15/demo/test_demo.py  --input-path="models/experimental/metal_BERT_large_15/demo/input_data.json"

This will run inference on the input data provided.
The model will first execute one time to fill up programm cache and then will proceed to execute inference on the provided data.

The model will print inference results on screen and will also report throughput and duration of compilation and inference.


Input file
----------

This model expects `JSON` input file with a list of 9 inputs, each consisting of context and quesiton.
Here is an example of the input file for only 2 inputs.

.. code-block:: json

    [
        {
            "context" : "Paragraph of text",
            "question" : "Question about the paragraph of text?"
        },
        {
            "context" : "Another paragraph of text",
            "question" : "Another question?"
        },
    ]
