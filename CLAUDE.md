# Introduction
We are working on productionize `models/tt_transformers`.

We refer to tt_transformers as TTT in this document. The current version of TTT is called TTTv1 and we are working on TTTv2.

# Motivations
Scaling problem to support 100+ models

Currently one change in TTTv1 has many consequences:
- Adding a new model or modifying an existing model's performance requires retesting every combination of all other models and hardware that we support
- Each contributor faces a Models x Platforms x Tests problem

This problem gets worse with each newly added model

# Goals
Single developer can add a new LLM model without tribal knowledge

TT-Transformers (TTT) code as a library
- Modular
- Composable
- Readable
- Reproducible
- Maintainable and releasable

TTTv1 is good, first achievement of the goals for 10+ models; now TTTv2 must do better to get to 100+ models

# Approach
![TTTv2 Approach](/home/gwang/tttv2.png) shows high level architecture of TTTv2.

## Key Design Principles

### Tightened Scope
- TTTv2 has a clearly defined boundary where only blocks overlapping with the TTTv2 circle are considered core parts of the project
- Model implementations themselves are **not** core parts of TTTv2, allowing for better separation of concerns

### Model Implementation Strategy
Model implementations (Llama/Qwen, Mistral, Gemma3, etc.) follow these principles:
- Not owned or maintained by TTTv2 team
- Implemented on basis of a specific release of TTTv2
- Can overwrite defaults to customize behavior
- Upgrades to TTTv2 (e.g., 2.0 → 2.1 following semantic versioning) do not require model implementations to upgrade

### Architecture Flow
The architecture follows a clear flow of information (as shown by arrows in the diagram):

1. **Model Conversion Layer**: Model implementations convert model-specific formats to TTTv2 format through copy/modify operations
2. **Standard Interfaces**:
   - Standard demos provide entry points
   - Standard vLLM generator handles model execution
   - TT HW config APIs manage hardware configurations
3. **ML Model Configuration**: Provides ML model config APIs
4. **Core TTT Modules**: depends on TTNN, these are the fundamental building blocks; minimize other dependencies
5. **Testing Infrastructure**: Unit tests for all key instances of TTT modules ensure reliability
6. **Debug Support**: Debug mode allows for detailed inspection and troubleshooting
7. **Performance Monitoring**: APIs to select unit tests to run, with accuracy and performance expectations for each module

This modular approach enables developers to add new LLM models without requiring deep tribal knowledge, while maintaining clear boundaries between the core library and model-specific implementations.

## Usage

Model implementations should:
- Import TTTv2 modules as needed
- Override defaults where necessary
- Implement model-specific logic separately
- Pin to specific TTTv2 version for stability

## Version Compatibility

TTTv2 follows semantic versioning:
- Major version changes (2.0 → 3.0): Breaking API changes
- Minor version changes (2.0 → 2.1): Backwards compatible features
- Patch version changes (2.0.0 → 2.0.1): Bug fixes only
