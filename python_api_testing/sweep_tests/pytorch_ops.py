import torch


def datacopy(x):
    return x


def recip(x):
    return torch.reciprocal(x)


def exp(x):
    return torch.exp(x)


def sqrt(x):
    return torch.sqrt(x)


def gelu(x):
    return torch.nn.functional.gelu(x)


def relu(x):
    return torch.nn.functional.relu(x)


def sigmoid(x):
    return torch.nn.functional.sigmoid(x)


def log(x):
    return torch.log(x)


def tanh(x):
    return torch.tanh(x)


def add(x, y):
    return torch.add(x, y)


def sub(x, y):
    return torch.sub(x, y)


def mul(x, y):
    return torch.mul(x, y)


def matmul(x, y):
    return torch.matmul(x, y)


def reduce_sum(x, dims=None, keepdim=True):
    return torch.sum(x, dims, keepdim)


def reduce_max(x, dims=None, keepdim=True):
    return torch.amax(x, dims, keepdim)


def flatten(x):
    return torch.flatten(x)


def transpose(x, dim0=-2, dim1=-1):
    return torch.transpose(x, dim0, dim1)
