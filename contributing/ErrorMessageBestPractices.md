# Guidelines for Writing Effective Error Messages ✍️
Clear and informative error messages are crucial for debugging and maintenance. A well-crafted error message can save hours of troubleshooting and make our codebase more user-friendly, especially for those less familiar with the system.

A well-written error message provides the following information to the user:
* What happened and why?
* What is the end result for the user?
* What can the user do to prevent it from happening again?

## Key Principles
### 1. Be Specific
Always include the actual values and conditions that caused the error. This helps to immediately identify the issue without needing to dig into the code.
Vague messages like "An error occurred" or "Invalid input" don’t help in identifying the root cause.

Instead of:
```cpp
TT_FATAL(input_shape.rank() == 3, "Invalid input tensor dimensions.");
```
Write:
```cpp
TT_FATAL(input_shape.rank() == 3, "Invalid input tensor: expected 3 dimensions, but found {}.", input_shape.rank());
```
### 2. Explain the Issue
Provide a brief explanation of why the error occurred or why the condition is important. This helps users understand the context of the error.

Instead of:
```cpp
TT_FATAL(!input_tensor_kv.has_value(), "KV tensor cannot be passed in when sharded.");
```
Write:
```cpp
TT_FATAL(!input_tensor_kv.has_value(), "Invalid operation: KV tensor should not be provided when the input tensor is sharded. Please ensure that the KV tensor is only used in non-sharded configurations.");
```

### 3. Include Relevant Information
Always include relevant variable values and context in the error message to aid in quick identification of the issue.
Omitting variable values or relevant details makes debugging harder.

Instead of:
```cpp
TT_FATAL(ptr != nullptr, "Pointer must not be null.");
```
Write:
```cpp
TT_FATAL(ptr != nullptr, "Failed to allocate memory: pointer is null.");
```
### 4. Make the Message Actionable
Ensure the error message provides clear guidance on what needs to be done to resolve the issue.
Stating what went wrong without providing guidance on how to fix it can be frustrating for users.

Instead of:
```cpp
TT_FATAL(head_size % TILE_WIDTH != 0, "Head size is invalid.");
```
Write:
```cpp
TT_FATAL(head_size % TILE_WIDTH != 0, "Invalid head size: {}. The head size must be a multiple of tile width ({}). Please adjust the dimensions accordingly.", head_size, TILE_WIDTH);
```

## Good Example
This message clearly states the problem, includes the actual value of head_size, and offers guidance on how to fix it.
```cpp
TT_FATAL(head_size % TILE_WIDTH == 0,
         "Invalid head size: {}. The head size must be a multiple of the tile width ({}). Please adjust the dimensions accordingly.",
         head_size, TILE_WIDTH);
```

## Style recommendations
* **Use simple, complete sentences.**
* **Use present tense** to describe current issues, and past tense for things that happened already.
* **Use active voice** when possible; passive voice is okay for describing errors.
* **Avoid using** ALL CAPS and exclamation points.
* **Clarify terms** by adding descriptors before them. For example,<br>instead of `Specify Axis when Merge is set to No`, say `Specify the Axis parameter when the Merge option is set to No.`
* **Don’t use the word "bad."** Be specific about what’s wrong. Instead of "Bad size," explain what size is needed.
* **Avoid the word "please."** It can make required actions sound optional.
* **Start your message with** the most important words that relate to the issue.
