from abc import ABC, abstractmethod

import numpy as np

from boltz.data.types import Structure


class StaticFilter(ABC):
    """Base class for structure filters."""

    @abstractmethod
    def filter(self, structure: Structure) -> np.ndarray:
        """Filter chains in a structure.

        Parameters
        ----------
        structure : Structure
            The structure to filter chains from.

        Returns
        -------
        np.ndarray
            The chains to keep, as a boolean mask.

        """
        raise NotImplementedError
