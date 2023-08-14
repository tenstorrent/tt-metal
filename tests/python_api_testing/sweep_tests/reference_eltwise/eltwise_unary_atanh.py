import matplotlib.pyplot as plt
import numpy as np
import torch


def custom_atanh_formula(x):
  # 0.5 * ln((1 + x) / (1 - x))
  return 0.5 * np.log((x+1) * (1/(1-x)))

x = np.linspace(-100, 100, 10000)

t_in = torch.from_numpy(x)
t_out = torch.atanh(t_in)
cust_out_formula = custom_atanh_formula(x)


plt.plot(x, t_out, "-b", label="torch atanh")
plt.plot(x, cust_out_formula, "+r", label="custom atanh formula")
plt.legend(loc="upper center")
plt.show()
