Converting torch Model to ttnn
###############################

.. note::
   This particular example only works on Grayskull.

   Not all converted models may be functional on all Tenstorrent hardware
   (Grayskull, Wormhole, or others). Functionality is on a case-by-case basis.

There are many ways to convert a torch model to ttnn.

This is the recommend approach:
    #. Re-writing torch model using functional torch APIs
    #. Converting operations of the functional torch model to ttnn operations
    #. Optimizing functional ttnn model

Step 1 - Rewriting the Model
****************************

Given a torch model, it can be rewritten using functional torch APIs.

For example, given the following torch model:

.. code-block:: python

    # From transformers.models.bert.modeling_bert.BertIntermediate

    import torch

    class BertIntermediate(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = torch.nn.Linear(config.hidden_size, config.intermediate_size)

        def forward(self, hidden_states):
            hidden_states = self.dense(hidden_states)
            hidden_states = torch.nn.functional.gelu(hidden_states)
            return hidden_states

Following TDD, the first step is to write a test for the model:

.. code-block:: python

    import pytest
    import torch
    import transformers

    import ttnn
    import torch_bert

    from models.utility_functions import torch_random
    from tests.ttnn.utils_for_testing import assert_with_pcc

    @pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("sequence_size", [384])
    def test_bert_intermediate(model_name, batch_size, sequence_size):
        torch.manual_seed(0)

        config = transformers.BertConfig.from_pretrained(model_name)
        model = transformers.models.bert.modeling_bert.BertIntermediate(config).eval()

        torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
        torch_output = model(torch_hidden_states) # Golden output

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model, # Function to initialize the model
            convert_to_ttnn=lambda *_: False, # Keep the weights as torch tensors
        )

        output = torch_bert.bert_intermediate(
            torch_hidden_states,
            parameters=parameters,
        )

        assert_with_pcc(torch_output, output, 0.9999)

And finally, the model can be rewritten using functional torch APIs to make the test pass:

.. code-block:: python

    # torch_bert.py

    def bert_intermediate(hidden_states, *, parameters):
        hidden_states = hidden_states @ parameters.dense.weight
        hidden_states = hidden_states + parameters.dense.bias
        hidden_states = torch.nn.functional.gelu(hidden_states)
        return hidden_states

.. note::

    ``parameters`` is a dictionary which sets its keys as its attributes, so both ``parameters["dense"]["weight"]`` and ``parameters.dense.weight`` are valid.

    The structure of ``parameters`` follows the structure of the model class.
    In this case, ``BertIntermediate`` has a single attribute ``dense``, so ``parameters`` has a single attribute ``dense``.
    And ``dense`` is a ``torch.nn.Linear`` object, so it in turn has two attributes ``weight`` and ``bias``.


Step 2 - Switching to ttnn Operations
*************************************

Starting off with the test:

.. code-block:: python

    import pytest
    import torch
    import transformers

    import ttnn
    import ttnn_bert

    from models.utility_functions import torch_random
    from tests.ttnn.utils_for_testing import assert_with_pcc

    @pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("sequence_size", [384])
    def test_bert_intermediate(device, model_name, batch_size, sequence_size):
        torch.manual_seed(0)

        config = transformers.BertConfig.from_pretrained(model_name)
        model = transformers.models.bert.modeling_bert.BertIntermediate(config).eval()

        torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1)
        torch_output = model(torch_hidden_states)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device, # Device to put the parameters on
        )

        hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        output = ttnn_bert.bert_intermediate(
            hidden_states,
            parameters=parameters,
        )
        output = ttnn.to_torch(output)

        assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.999)

Then implementing the function using ttnn operations:

.. code-block:: python

    # ttnn_bert.py

    import ttnn

    def bert_intermediate(
        hidden_states,
        *,
        parameters,
    ):
        output = hidden_states @ parameters.dense.weight
        output = output + parameters.dense.bias
        output = ttnn.gelu(output)
        return output

Step 3 - Optimizing the Model
*****************************

Starting off with the test:

.. code-block:: python

    import pytest
    import torch
    import transformers

    import ttnn
    import ttnn_bert

    from models.utility_functions import torch_random
    from tests.ttnn.utils_for_testing import assert_with_pcc

    @pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("sequence_size", [384])
    def test_bert_intermediate(device, model_name, batch_size, sequence_size):
        torch.manual_seed(0)

        config = transformers.BertConfig.from_pretrained(model_name)
        model = transformers.models.bert.modeling_bert.BertIntermediate(config).eval()

        torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1)
        torch_output = model(torch_hidden_states)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=device, # Device to put the parameters on
            custom_preprocessor=ttnn_bert.custom_preprocessor, # Use custom_preprocessor to set ttnn.bfloat8_b data type for the weights and biases
        )

        hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        output = ttnn_bert.bert_intermediate(
            hidden_states,
            parameters=parameters,
        )
        output = ttnn.to_torch(output)

        assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.999)

And the optimized model can be something like this:

.. code-block:: python

    # ttnn_optimized_bert.py

    import ttnn
    import transformers

    def custom_preprocessor(model, name):

        parameters = {}
        if isinstance(model, transformers.models.bert.modeling_bert.BertIntermediate):
            parameters["weight"] = ttnn.model_preprocessing.preprocess_linear_weight(model.weight, dtype=ttnn.bfloat8_b)
            parameters["bias"] = ttnn.model_preprocessing.preprocess_linear_bias(model.bias, dtype=ttnn.bfloat8_b)

        return parameters

    def bert_intermediate(
        hidden_states,
        *,
        parameters,
        num_cores_x,
    ):
        batch_size, *_ = hidden_states.shape

        num_cores_x = 12
        output = ttnn.linear(
            hidden_states,
            ff1_weight,
            bias=ff1_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG, # Put the output into local core memory
            core_grid=(batch_size, num_cores_x), # Specify manual core grid to get the best possible performance
            activation="gelu", # Fuse Gelu
        )
        return True

More examples
*************

Additional examples can be found in `the integration tests <https://github.com/tenstorrent/tt-metal/tree/main/tests/ttnn/integration_tests>`_.
