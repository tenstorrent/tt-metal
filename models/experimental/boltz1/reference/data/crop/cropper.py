from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from boltz.data.types import Tokenized


class Cropper(ABC):
    """Abstract base class for cropper."""

    @abstractmethod
    def crop(
        self,
        data: Tokenized,
        max_tokens: int,
        random: np.random.RandomState,
        max_atoms: Optional[int] = None,
        chain_id: Optional[int] = None,
        interface_id: Optional[int] = None,
    ) -> Tokenized:
        """Crop the data to a maximum number of tokens.

        Parameters
        ----------
        data : Tokenized
            The tokenized data.
        max_tokens : int
            The maximum number of tokens to crop.
        random : np.random.RandomState
            The random state for reproducibility.
        max_atoms : Optional[int]
            The maximum number of atoms to consider.
        chain_id : Optional[int]
            The chain ID to crop.
        interface_id : Optional[int]
            The interface ID to crop.

        Returns
        -------
        Tokenized
            The cropped data.

        """
        raise NotImplementedError
