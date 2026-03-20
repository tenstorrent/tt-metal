import ttnn
import ttml
from ttml.autograd import Function


class Exp(Function):
    @staticmethod
    def forward(ctx, x):
        result = ttnn.exp(x.get_value())
        out = ttml.autograd.create_tensor(result)
        ctx.save_for_backward(out)  # save exp(x) for backward
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (exp_x,) = ctx.saved_tensors
        # d/dx exp(x) = exp(x)

        return ttnn.multiply(grad_output, exp_x.get_value())


class Clip(Function):
    @staticmethod
    def forward(ctx, x, lo, hi):
        val = x.get_value()
        clipped = ttnn.clip(val, lo, hi)

        # mask: 1.0 where lo <= x <= hi, 0.0 where clipped
        mask = ttnn.multiply(ttnn.ge(val, lo), ttnn.le(val, hi))
        ctx.mask = mask
        return clipped

    @staticmethod
    def backward(ctx, grad_output):
        return ttnn.multiply(grad_output, ctx.mask)


class Min(Function):
    @staticmethod
    def forward(ctx, a, b):
        a_val = a.get_value()
        b_val = b.get_value()
        result = ttnn.minimum(a_val, b_val)
        # mask: 1.0 where a <= b (a is the min), 0.0 where b < a
        a_wins = ttnn.le(a_val, b_val)
        ctx.a_wins = a_wins
        return result

    @staticmethod
    def backward(ctx, grad_output):
        a_wins = ctx.a_wins
        # b_wins is the complement
        b_wins = ttnn.subtract(ttnn.ones_like(a_wins), a_wins)
        grad_a = ttnn.multiply(grad_output, a_wins)
        grad_b = ttnn.multiply(grad_output, b_wins)
        return grad_a, grad_b
