Adding New ttnn Operation
#########################


C++ Implementation
------------------


Add `tt_eager/tt_dnn/op_library/<new_operation>/<new_operation>.hpp`:

.. code-block:: cpp

    #pragma once

    #include <optional>

    #include "tensor/tensor.hpp"
    #include "tt_dnn/op_library/operation.hpp"

    namespace tt {
    namespace tt_metal {

    struct <NewOperation> {
        bool some_arg;

        // These methods are needed if the operation takes in input tensor and produces output tensors
        void validate(const std::vector<Tensor> &input_tensors) const;
        std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
        std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
        operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;

        static constexpr auto attribute_names = std::make_tuple();
        const auto attribute_values() const {
            return std::make_tuple();
        }
    };

    Tensor <new_operation>(const Tensor &input_tensor, bool some_arg);

    }  // namespace tt_metal
    }  // namespace tt


.. note:

    If you need optional input tensors or would like to pass in optional output tensors, then refer to :doc:`Operations </ttnn/dependencies/tt_lib>` for how to write ops that use them


Add `tt_eager/tt_dnn/op_library/<new_operation>/<new_operation>.cpp`:

.. code-block:: cpp

    #include "tt_metal/host_api.hpp"
    #include "tt_dnn/op_library/run_operation.hpp"

    namespace tt {
    namespace tt_metal {


    void <NewOperation>::validate(const std::vector<Tensor> &input_tensors) const {
        ...
    }

    std::vector<Shape> <NewOperation>::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
        std::vector<Shape> output_shapes = ...;
        return output_shapes;
    }

    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const {
        std::vector<Tensor> output_tensors = ...;
        return output_tensors;
    }

    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
        Program program = ...;
        return operation::ProgramWithCallbacks{program};
    }

    };

    Tensor <new_operation>(const Tensor &input_tensor, bool some_arg) {
        std::vector<Tensor> input_tensors = {input_tensor};
        std::vector<Tensor> output_tensors operation::run(DeviceOperation(<NewOperation>{some_arg}, {input_tensor}));
        return output_tensors[0];
    }

    }  // namespace tt_metal
    }  // namespace tt


Add pybindings
--------------

In `tt_eager/tt_lib/csrc/tt_lib_bindings_tensor.cpp`, add the following lines

.. code-block:: cpp

    m_tensor.def("<new_operation>", &<new_operation>, py::arg("input_tensor").noconvert(), py::arg("some_arg").noconvert(), R"doc(
        <NewOperation> runs new operation on input tensor.

        .. csv-table::
            :header: "Argument", "Description", "Data type", "Valid range", "Required"

            "input_tensor", "Input tensor", "Tensor", "Tensor of shape [W0, Z0, Y0, X0]", "Yes"
            "some_arg", "Some arg", "bool", "Some arg to do some stuff in new operation", "Yes"
    )doc");



Adding a unit test
------------------

Add `tests/ttnn/unit_tests/ttl/test_<new_operation>.py`:

.. code-block:: python

    import pytest
    import torch
    import ttnn

    from tests.ttnn.utils_for_testing import assert_with_pcc

    @pytest.mark.parametrize("height", [32])
    @pytest.mark.parametrize("width", [32])
    def test_<new_operation>(device, height, width):
        torch.manual_seed(0)

        torch_input_tensor = torch.rand(1, 1, height, width)
        torch_output_tensor = torch.exp(torch_input_tensor)

        input_tensor = ttnn.from_torch(torch_input_tensor, device=device)
        output_tensor = ttnn.experimental.tensor.<new_operation>(input_tensor)

        output_tensor = ttnn.to_torch(output_tensor)

        assert_with_pcc(torch_output_tensor, output_tensor)



Adding a sweep test
-------------------

Add `tests/ttnn/sweep_tests/sweeps/ttl_<new_operation>.py`:

.. code-block:: python

    from typing import Optional, Tuples
    import torch
    import ttnn
    from tests.ttnn.utils_for_testing import check_with_pcc


    parameters = {
        "height": [384, 1024],
        "width": [1024, 4096],
    }


    def skip(**_) -> Tuple[bool, Optional[str]]:
        return False, None


    def is_expected_to_fail(**_) -> Tuple[bool, Optional[str]]:
        return False, None


    def run(
        height,
        width,
        *,
        device,
    ) -> Tuple[bool, Optional[str]]:

        torch_input_tensor = torch.rand(1, 1, height, width)
        torch_output_tensor = torch.exp(torch_input_tensor)

        input_tensor = ttnn.from_torch(torch_input_tensor, device=device)
        output_tensor = ttnn.experimental.tensor.<new_operation>(input_tensor)

        output_tensor = ttnn.to_torch(output_tensor)

        assert_with_pcc(torch_output_tensor, output_tensor)
