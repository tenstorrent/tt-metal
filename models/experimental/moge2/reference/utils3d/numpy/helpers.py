# decorator
import numpy as np
from numbers import Number
import inspect
from functools import wraps
from typing import *
from types import EllipsisType
from ..helpers import suppress_traceback

__all__ = [
    'toarray',
    'batched',
]

P = ParamSpec("P")  
R = TypeVar("R")


def toarray(*args_dtypes: Union[np.dtype, str, None], _others: Union[np.dtype, str] = None, **kwargs_dtypes: Union[np.dtype, str]) -> Callable[[Callable[P, R]], Callable[P, R]]:
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
            args = tuple(
                np.array(x, inputs[dtype].dtype if isinstance(dtype, str) else dtype) 
                if isinstance(x, (Number, list, tuple)) \
                    and (dtype := dtypes_dict.get(argnames[i], _others)) is not None \
                else x
                for i, x in enumerate(args)
            )
            kwargs = {
                k: np.array(x, inputs[dtype].dtype if isinstance(dtype, str) else dtype) 
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
            batch_shape = np.broadcast_shapes(*(
                x.shape[:x.ndim - dim] 
                for x, dim in zip((*args, *kwargs.values()), (*args_dim, *kwargs_dim.values())) 
                if isinstance(x, np.ndarray) and dim is not None
            ))
            # Broadcast and flatten batch dimensions
            args = tuple(
                np.broadcast_to(x, (*batch_shape, *x.shape[x.ndim - dim:])).reshape((-1, *x.shape[x.ndim - dim:]))
                if isinstance(x, np.ndarray) and dim is not None else x
                for x, dim in zip(args, args_dim)
            )
            kwargs = {
                k: np.broadcast_to(x, (*batch_shape, *x.shape[x.ndim - dim:])).reshape((-1, *x.shape[x.ndim - dim:]))
                if isinstance(x, np.ndarray) and (dim := kwargs_dim[k]) is not None else x
                for k, x in kwargs.items()
            }
            # Call function
            result = func(*args, **kwargs)
            # Restore batch shape
            if isinstance(result, tuple):
                result = tuple(
                    x.reshape((*batch_shape, *x.shape[1:])) if isinstance(x, np.ndarray) else x
                    for x in result
                )
            elif isinstance(result, np.ndarray):
                result = result.reshape((*batch_shape, *result.shape[1:]))
            return result
        return wrapper
    return decorator
