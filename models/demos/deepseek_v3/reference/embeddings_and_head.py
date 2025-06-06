# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch DeepSeek Embeddings and Language Model Head modules."""

from torch import nn


class DeepseekV3Embeddings(nn.Module):
    """Token embeddings for DeepseekV3 model

    Pseudo-code:
    ```
    def forward(input_ids):
        # Input shape: [batch_size, seq_len] (int64 token indices)
        # Output shape: [batch_size, seq_len, hidden_size=7168]

        # Lookup embeddings for each token ID
        embeddings = embed_tokens[input_ids]  # [batch, seq, 7168]

        return embeddings
    ```

    Shape Examples:
    - Prefill: [batch=1, seq=100] -> [1, 100, 7168]
    - Decode:  [batch=1, seq=1] -> [1, 1, 7168]

    Notes:
    - vocab_size = 129280
    - Padding tokens (padding_idx) are mapped to zero embeddings
    """

    def __init__(self, vocab_size, hidden_size, padding_idx):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx)

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


class DeepseekV3LMHead(nn.Module):
    """Language Model Head for DeepseekV3 model

    Pseudo-code:
    ```
    def forward(hidden_states):
        # Input shape: [batch_size, seq_len, hidden_size=7168]
        # Output shape: [batch_size, seq_len, vocab_size=129280]

        # Linear projection to vocabulary size
        logits = lm_head(hidden_states)  # [batch, seq, 129280]

        # Cast to float32 for numerical stability
        logits = logits.float()

        return logits
    ```

    Shape Examples:
    - Prefill: [batch=1, seq=100, hidden=7168] -> [1, 100, 129280]
    - Decode:  [batch=1, seq=1, hidden=7168] -> [1, 1, 129280]

    Notes:
    - No bias in the linear layer
    - Output is cast to float32 for stability in loss computation
    """

    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states):
        # Cast to float for numerical stability
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        return logits

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings


class DeepseekV3EmbeddingsTTNN(nn.Module):
    """TTNN implementation of Token embeddings for DeepseekV3 model

    Pseudo-code:
    ```
    def forward(input_ids):
        # Input shape: [1, 1, batch_size, seq_len] (int32 token indices)
        # Output shape: [1, 1, batch_size*seq_len, 7168]

        # Reshape to 1D for embedding lookup
        input_ids_1d = input_ids.reshape([1, 1, 1, batch_size*seq_len])

        # Lookup embeddings for each token ID
        embeddings = ttnn.embedding(input_ids_1d, embed_tokens, layout=ttnn.TILE_LAYOUT)
        # Output: [1, 1, batch_size*seq_len, 7168]

        return embeddings
    ```

    Shape Examples:
    - Prefill: [1, 1, 1, SEQ_LEN] -> [1, 1, SEQ_LEN, 7168]
    - Decode:  [1, 1, BATCH_SIZE, 1] -> [1, 1, BATCH_SIZE, 7168]

    Notes:
    - vocab_size = 129280
    - TTNN embedding expects flattened input_ids
    - Padding tokens (padding_idx) handled by having zero embeddings in weight matrix
    - For decode mode, input is [1, 1, BATCH_SIZE, 1] and gets reshaped
    """

    def __init__(self, vocab_size, hidden_size, padding_idx):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

        # Embedding weights will be loaded as ttnn tensor on device
        self.embed_tokens = None  # Will be set during weight loading

    def forward(self, input_ids, memory_config=None):
        """
        Args:
            input_ids: TTNN tensor of shape [1, 1, BATCH_SIZE, SEQ_LEN] (int32)
                      For prefill: [1, 1, 1, SEQ_LEN]
                      For decode: [1, 1, BATCH_SIZE, 1]
            memory_config: Optional memory configuration

        Returns:
            TTNN tensor of shape [1, 1, BATCH_SIZE*SEQ_LEN, 7168]
            For prefill: [1, 1, SEQ_LEN, 7168]
            For decode: [1, 1, BATCH_SIZE, 7168]
        """
        # Get shape for reshape
        batch_size = input_ids.shape[2]
        seq_len = input_ids.shape[3]

        # Reshape to 1D for embedding lookup
        input_ids_1d = ttnn.reshape(input_ids, [1, 1, 1, batch_size * seq_len])

        # Perform embedding lookup
        embeddings = ttnn.embedding(
            input_ids_1d, self.embed_tokens, layout=ttnn.TILE_LAYOUT, memory_config=memory_config
        )

        # Output shape: [1, 1, batch_size*seq_len, hidden_size]
        # For decode with batch: [1, 1, BATCH_SIZE, 7168]
        # For prefill: [1, 1, SEQ_LEN, 7168]
        return embeddings


class DeepseekV3LMHeadTTNN(nn.Module):
    """TTNN implementation of Language Model Head for DeepseekV3 model

    Pseudo-code:
    ```
    def forward(hidden_states):
        # Input shape: [1, 1, batch_size*seq_len, 7168]
        # Output shape: [1, 1, batch_size*seq_len, 129280]

        # Linear projection to vocabulary size
        logits = ttnn.linear(hidden_states, lm_head_weight,
                           memory_config=memory_config,
                           dtype=ttnn.bfloat16)  # [1, 1, batch*seq, 129280]

        # Note: Cast to float32 would happen on host after to_torch()
        return logits
    ```

    Shape Examples:
    - Prefill: [1, 1, SEQ_LEN, 7168] -> [1, 1, SEQ_LEN, 129280]
    - Decode:  [1, 1, BATCH_SIZE, 7168] -> [1, 1, BATCH_SIZE, 129280]

    Notes:
    - No bias in the linear layer
    - Output should be cast to float32 on host for stability in loss computation
    - Large vocabulary size (129280) may require special memory handling
    """

    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # LM head weight will be loaded as ttnn tensor on device
        self.lm_head_weight = None  # Will be set during weight loading

    def forward(self, hidden_states, memory_config=None, compute_kernel_config=None):
        """
        Args:
            hidden_states: TTNN tensor of shape [1, 1, BATCH_SIZE*SEQ_LEN, 7168]
            memory_config: Optional memory configuration (likely DRAM for large vocab)
            compute_kernel_config: Optional compute configuration

        Returns:
            TTNN tensor of shape [1, 1, BATCH_SIZE*SEQ_LEN, 129280]
        """
        logits = ttnn.linear(
            hidden_states,
            self.lm_head_weight,
            bias=None,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
        )

        # Note: Conversion to float32 happens after to_torch() on host
        return logits

    def forward_split(self, hidden_states, memory_config=None, compute_kernel_config=None):
        """
        Split computation version for very large vocabulary
        Splits vocab dimension to fit in memory
        """
        # Split vocab into chunks that fit in L1/DRAM
        vocab_chunk_size = 32768  # Example chunk size
        num_chunks = (self.vocab_size + vocab_chunk_size - 1) // vocab_chunk_size

        chunks = []
        for i in range(num_chunks):
            start_idx = i * vocab_chunk_size
            end_idx = min((i + 1) * vocab_chunk_size, self.vocab_size)

            # Slice weight matrix
            weight_chunk = self.lm_head_weight[:, start_idx:end_idx]

            # Compute chunk
            chunk_logits = ttnn.linear(
                hidden_states,
                weight_chunk,
                bias=None,
                memory_config=memory_config,
                compute_kernel_config=compute_kernel_config,
                dtype=ttnn.bfloat16,
            )
            chunks.append(chunk_logits)

        # Concatenate chunks along vocab dimension
        logits = ttnn.concat(chunks, dim=3, memory_config=memory_config)

        return logits
