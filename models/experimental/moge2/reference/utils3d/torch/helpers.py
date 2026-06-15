# decorator
import torch
from torch import Tensor
from numbers import Number
import inspect
from typing import *
from functools import wraps
from ..helpers import suppress_traceback


__all__ = [
    'toarray',
    'batched',
]

P = ParamSpec("P")  
R = TypeVar("R")

def totensor(
    *args_dtypes: Union[torch.dtype, Tuple[torch.dtype, torch.device], str, None], 
    _others: Union[torch.dtype, str] = None, 
    **kwargs_dtypes: Union[torch.dtype, Tuple[torch.dtype, torch.device], str]
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator generator that converts non-array arguments to array of specified default dtype.
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        argnames = list(inspect.signature(func).parameters.keys())
        dtypes_dict = {
            **dict(zip(argnames, args_dtypes)),
            **kwargs_dtypes
        }
        @wraps(func)
        @suppress_traceback
        def wrapper(*args, **kwargs):
            inputs = {
                **{argnames[i]: x for i, x in enumerate(args)},
                **kwargs
            }
            if len(input_devices := tuple(x.device for x in inputs.values() if isinstance(x, Tensor))) > 0:
                device = input_devices[0]
            else:
                device = None
            args = tuple(
                torch.tensor(x).to(device, getattr(inputs[dtype], 'dtype', None) if isinstance(dtype, str) else dtype)
                if isinstance(x, (Number, list, tuple)) \
                    and (dtype := dtypes_dict.get(argnames[i], _others)) is not None \
                else x
                for i, x in enumerate(args)
            )
            kwargs = {
                k: torch.tensor(x).to(device, getattr(inputs[dtype], 'dtype', None) if isinstance(dtype, str) else dtype)
                if isinstance(x, (Number, list, tuple)) \
                    and (dtype := dtypes_dict.get(k, _others)) is not None \
                else x
                for k, x in kwargs.items()
            }
            return func(*args, **kwargs)
        return wrapper
    return decorator


def batched(*args_dims: Union[int, None], _others: Union[int, None] = None, **kwargs_dims: Union[int, None]) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator generator that extends a function's input and out batch dimensions.
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        argnames = list(inspect.signature(func).parameters.keys())
        dims_dict = {
            **dict(zip(argnames, args_dims)),
            **kwargs_dims
        }
        @wraps(func)
        @suppress_traceback
        def wrapper(*args, **kwargs):
            args = list(args)
            # Get arguments non-batch dimensions
            args_dim = tuple(dims_dict.get(argname, _others) for argname in argnames[:len(args)])
            kwargs_dim = {k: dims_dict.get(k, _others) for k in kwargs}
            # Find the common batch shape
            batch_shape = torch.broadcast_shapes(*(
                x.shape[:x.ndim - dim] 
                for x, dim in zip((*args, *kwargs.values()), (*args_dim, *kwargs_dim.values())) 
                if isinstance(x, Tensor) and dim is not None
            ))
            # Broadcast and flatten batch dimensions
            args = tuple(
                torch.broadcast_to(x, (*batch_shape, *x.shape[x.ndim - dim:])).reshape((-1, *x.shape[x.ndim - dim:]))
                if isinstance(x, Tensor) and dim is not None else x
                for x, dim in zip(args, args_dim)
            )
            kwargs = {
                k: torch.broadcast_to(x, (*batch_shape, *x.shape[x.ndim - dim:])).reshape((-1, *x.shape[x.ndim - dim:]))
                if isinstance(x, Tensor) and (dim := kwargs_dim[k]) is not None else x
                for k, x in kwargs.items()
            }
            # Call function
            result = func(*args, **kwargs)
            # Restore batch shape
            if isinstance(result, tuple):
                result = tuple(
                    x.reshape((*batch_shape, *x.shape[1:])) if isinstance(x, Tensor) else x
                    for x in result
                )
            elif isinstance(result, Tensor):
                result = result.reshape((*batch_shape, *result.shape[1:]))
            return result
        return wrapper
    return decorator
