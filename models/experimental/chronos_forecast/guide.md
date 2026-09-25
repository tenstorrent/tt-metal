# Guide
This guide is structured to provide clear and concise information, making it easy for readers to understand and apply the concepts discussed.
Will go through the model archecture, the precision, testing, evaluation metrics, pratical applications, along with the optimzations done with this model and its configuration modes

## Model Archecture
Chronos comes in different variants, mainly the chronos2 and chronos bolt(smaller version), however this directory uses chronos2.
The forward pass of the model is shown in [`Chronos2Model.forward`](reference/chronos2/model.py#L637-L770)
The main blocks of the model are:
- **input layer**: input patch embedding + positional encoding
```
hidden = ReLU(input_projection(x))
output = output_projection(hidden)
skip = residual_projection(x)
embedding = output + skip
```

- **encoder block**: N x blocks of time self attention + group self attention + feed forward network
- chronos2 N = 12
```
x = x + time_attention(x)
x = x + group_attention(x)
x = x + feed_forward(x)
```
- **rmsnorm**
- **output layer**: linear projection to the output dimension, will have another layer that undoes the norm
