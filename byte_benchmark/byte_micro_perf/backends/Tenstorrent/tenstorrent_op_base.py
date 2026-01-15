"""
Base class for Tenstorrent operations.
Provides common functionality to avoid pin_memory issues on systems without CUDA.
"""


class TenstorrentOpMixin:
    """
    Mixin class for Tenstorrent operations.
    Overrides tensor creation to avoid pin_memory calls that require CUDA.
    """

    def _create_in_out_tensors(
        self,
        instance_num,
        create_inputs=True,
        create_outputs=True,
    ):
        """
        Override tensor creation to avoid pin_memory issues on systems without CUDA.
        Tenstorrent handles memory management via tt-metal, not CUDA pinned memory.
        """
        all_tensor_list = []

        # create first instance
        first_tensor_mapping = {}
        if create_inputs:
            for key, value in self.input_tensor_info.items():
                first_tensor_mapping[key] = value.creator(size=value.shape, dtype=value.dtype, device=value.device)
                # Don't pin memory for Tenstorrent
                # Memory management is handled by tt-metal, not CUDA
                # if value.device == "cpu":
                #     first_tensor_mapping[key] = first_tensor_mapping[key].pin_memory()

        if create_outputs:
            for key, value in self.output_tensor_info.items():
                first_tensor_mapping[key] = value.creator(size=value.shape, dtype=value.dtype, device=value.device)
                # Don't pin memory for Tenstorrent
                # Memory management is handled by tt-metal, not CUDA
                # if value.device == "cpu":
                #     first_tensor_mapping[key] = first_tensor_mapping[key].pin_memory()

        all_tensor_list.append(first_tensor_mapping)

        # clone following instances (same as core implementation)
        for _ in range(instance_num - 1):
            tensor_mapping = {}
            for key, value in first_tensor_mapping.items():
                tensor_mapping[key] = value.clone()
            all_tensor_list.append(tensor_mapping)

        return all_tensor_list
