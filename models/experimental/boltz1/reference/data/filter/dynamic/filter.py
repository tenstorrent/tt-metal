from abc import ABC, abstractmethod

from boltz.data.types import Record


class DynamicFilter(ABC):
    """Base class for data filters."""

    @abstractmethod
    def filter(self, record: Record) -> bool:
        """Filter a data record.

        Parameters
        ----------
        record : Record
            The object to consider filtering in / out.

        Returns
        -------
        bool
            True if the data passes the filter, False otherwise.

        """
        raise NotImplementedError
